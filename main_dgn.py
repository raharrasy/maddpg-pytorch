import argparse
import torch
import time
import lbforaging
import gym
import os
import random
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env_lbf
from utils.make_env import make_env_wolf
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.dgn import DGN
from gym.vector import AsyncVectorEnv

USE_CUDA = False  # torch.cuda.is_available()

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def make_parallel_env_lbf(
        env_id, n_rollout_threads, seed=1285, effective_max_num_players=3,
        with_shuffle=True, multi_agent=True) :
    if n_rollout_threads == 1:
        return DummyVecEnv([
            make_env_lbf("MARL-"+env_id, seed=seed,
                effective_max_num_players=effective_max_num_players,
                with_shuffle=with_shuffle,
                multi_agent=multi_agent)()
        ])
    else:
        return SubprocVecEnv([
            make_env_lbf(
                "MARL-"+env_id, seed=seed+i,
                effective_max_num_players=effective_max_num_players,
                with_shuffle=with_shuffle,
                multi_agent=multi_agent)
                for i in range(n_rollout_threads)
        ])


def make_parallel_env_wolf(
        env_id, n_rollout_threads, seed=1285,
        close_penalty=0.5, implicit_max_player_num=3, max_player_num=5) :
    if n_rollout_threads == 1:
        return DummyVecEnv([
            make_env_wolf("MARL-"+env_id, seed=seed,
                close_penalty=close_penalty,
                implicit_max_player_num=implicit_max_player_num,
                max_player_num= max_player_num)()
        ])
    else:
        return SubprocVecEnv([
            make_env_wolf(
                "MARL-"+env_id, seed=seed+i,
                close_penalty=close_penalty,
                implicit_max_player_num=implicit_max_player_num,
                max_player_num= max_player_num)
                for i in range(n_rollout_threads)
        ])

def make_env_adhoc(env_id, rank,  seed=1285, effective_max_num_players=3, with_shuffle=True):
    def _init():
        env = gym.make(
            "Adhoc-"+env_id, seed=seed + rank,
            effective_max_num_players=effective_max_num_players,
            init_num_players=effective_max_num_players,
            with_shuffle=with_shuffle
        )
        return env

    return _init

def make_env_adhoc_wolf(env_id, rank, num_players, seed=1285, close_penalty=0.5, implicit_max_player_num=3, with_shuffling=True):
    def _init():
        env = gym.make(
            "Adhoc-"+env_id, seed=seed + rank, num_players=num_players, close_penalty=close_penalty,
            implicit_max_player_num=implicit_max_player_num,  with_shuffling=with_shuffling
        )
        return env

    return _init

def obs_adhoc_postprocess_wolf(obs_teammate, obs_prey):
    num_agents = obs_teammate.shape[1]//2
    dgn_obs = np.asarray([[np.roll(ob,-2*i)for i in range((len(ob)//2))]for ob in obs_teammate])
    repeated_obs_prey = np.repeat(np.expand_dims(obs_prey, axis=1), num_agents, axis=1)
    obs = np.concatenate([dgn_obs, repeated_obs_prey, torch.ones([dgn_obs.shape[0], dgn_obs.shape[1], 1])], axis = -1)

    obs[obs[:,:,0] == -1] = -1 * np.ones(obs.shape[2])
    return torch.tensor(obs)

def convert_action(one_hot_acts):
    return np.argmax(np.concatenate([
        np.expand_dims(act, axis=0) for act in one_hot_acts
    ], axis=0), axis=-1).tolist()

def compute_average_reward(obs, rews):
    n_agents = np.sum(obs[:,:,-1] != -1, axis=-1)
    total_rewards = np.sum(rews, axis=-1)
    return [(a+0.0)/b if b!=0 else 0 for a,b in zip(total_rewards, n_agents)]

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env_wolf(
        config.env_id, config.n_rollout_threads,
        config.seed,
        config.close_penalty,
        config.num_players_train
    )

    marl_env_train = make_parallel_env_wolf(
        config.env_id, config.n_rollout_threads,
        config.seed,
        config.close_penalty,
        config.num_players_train
    )
    marl_env_test = make_parallel_env_wolf(
        config.env_id, config.n_rollout_threads,
        config.seed,
        config.close_penalty,
        config.num_players_test
    )
    env_train_adhoc = AsyncVectorEnv(
        [
            make_env_adhoc_wolf('wolfpack-v5', i, config.num_players_train, 2000, config.close_penalty)
            for i in range(config.n_rollout_threads)
        ]
    )

    env_eval_adhoc = AsyncVectorEnv(
        [
            make_env_adhoc_wolf(
                'wolfpack-v5', i, config.num_players_train, 2000, config.close_penalty,
                implicit_max_player_num=config.num_players_test
            )
            for i in range(config.n_rollout_threads)
        ]
    )

    dgn = DGN.init_from_env(env, alg=config.alg, temp=config.temp,
                            tau=config.tau, lr_crit=config.lr_crit,
                            reg=config.reg)
    replay_buffer = ReplayBuffer(config.buffer_length, dgn.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])

    dgn.save(model_dir / curr_run / "params_0.pt")

    # Train env
    avgs = []
    dgn.prep_rollouts(device='cpu')
    num_dones, per_worker_rew = [0] * config.n_rollout_threads, [0] * config.n_rollout_threads
    obs = marl_env_train.reset()
    while (all([k < config.eval_eps for k in num_dones])):
        torch_obs = Variable(torch.tensor(obs),
                 requires_grad=False).float()
        # get actions as torch Variables
        agent_actions = dgn.step(torch_obs)
        # Env step
        next_obs, rewards, dones, infos = marl_env_train.step(agent_actions)
        average_step_rews = compute_average_reward(obs, rewards)
        dones = [any(x) for x in dones]

        per_worker_rew = [k + l for k, l in zip(per_worker_rew, average_step_rews)]
        obs = next_obs

        for idx, flag in enumerate(dones):
            if flag:
                if num_dones[idx] < config.eval_eps:
                    num_dones[idx] += 1
                    avgs.append(per_worker_rew[idx])
                per_worker_rew[idx] = 0

    print("Finished marl train with rewards " + str((sum(avgs) + 0.0) / len(avgs)))
    marl_env_train.close()
    logger.add_scalar('Rewards/marl_train_set', (sum(avgs) + 0.0) / len(avgs), 0)

    # Test env
    avgs = []
    dgn.prep_rollouts(device='cpu')
    num_dones, per_worker_rew = [0] * config.n_rollout_threads, [0] * config.n_rollout_threads
    obs = marl_env_test.reset()
    while (all([k < config.eval_eps for k in num_dones])):
        torch_obs = Variable(torch.tensor(obs),
                 requires_grad=False).float()
        # get actions as torch Variables
        agent_actions = dgn.step(torch_obs)

        # Env step
        next_obs, rewards, dones, infos = marl_env_test.step(agent_actions)
        average_step_rews = compute_average_reward(obs, rewards)
        dones = [any(x) for x in dones]

        per_worker_rew = [k + l for k, l in zip(per_worker_rew, average_step_rews)]
        obs = next_obs

        for idx, flag in enumerate(dones):
            if flag:
                if num_dones[idx] < config.eval_eps:
                    num_dones[idx] += 1
                    avgs.append(per_worker_rew[idx])
                per_worker_rew[idx] = 0

    print("Finished marl eval with rewards " + str((sum(avgs) + 0.0) / len(avgs)))
    marl_env_test.close()
    logger.add_scalar('Rewards/marl_eval_set', (sum(avgs) + 0.0) / len(avgs), 0)

    avgs = []
    dgn.prep_rollouts(device='cpu')
    num_dones, per_worker_rew = [0] * config.n_rollout_threads, [0] * config.n_rollout_threads
    obs = env_train_adhoc.reset()
    while (all([k < config.eval_eps for k in num_dones])):
        torch_obs = obs_adhoc_postprocess_wolf(obs['teammate_location'], obs['opponent_info'])
        # Agent computes actions
        agent_actions = dgn.indiv_step(torch_obs)
        # Env step
        next_obs, rewards, dones, infos = env_train_adhoc.step(agent_actions)
        per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
        obs = next_obs
        for idx, flag in enumerate(dones):
            if flag:
                if num_dones[idx] < config.eval_eps:
                    num_dones[idx] += 1
                    avgs.append(per_worker_rew[idx])
                per_worker_rew[idx] = 0

    print("Finished ad hoc train with rewards " + str((sum(avgs) + 0.0) / len(avgs)))
    env_train_adhoc.close()
    logger.add_scalar('Rewards/adhoc_train', (sum(avgs) + 0.0) / len(avgs), 0)

    avgs = []
    dgn.prep_rollouts(device='cpu')
    num_dones, per_worker_rew = [0] * config.n_rollout_threads, [0] * config.n_rollout_threads
    obs = env_eval_adhoc.reset()
    while (all([k < config.eval_eps for k in num_dones])):
        torch_obs = obs_adhoc_postprocess_wolf(obs['teammate_location'], obs['opponent_info'])
        # Agent computes actions
        agent_actions = dgn.indiv_step(torch_obs)
        # Env step
        next_obs, rewards, dones, infos = env_eval_adhoc.step(agent_actions)
        per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
        obs = next_obs
        for idx, flag in enumerate(dones):
            if flag:
                if num_dones[idx] < config.eval_eps:
                    num_dones[idx] += 1
                    avgs.append(per_worker_rew[idx])
                per_worker_rew[idx] = 0

    print("Finished ad hoc test with rewards " + str((sum(avgs) + 0.0) / len(avgs)))
    env_eval_adhoc.close()
    logger.add_scalar('Rewards/adhoc_test', (sum(avgs) + 0.0) / len(avgs), 0)
    t = 0

    num_episode = config.max_num_steps // config.eps_length
    for ep_i in range(num_episode):
        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        dgn.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        dgn.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        #maddpg.reset_noise()

        for et_i in range(config.eps_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = Variable(torch.tensor(obs),
                                  requires_grad=False).float()
            # get actions as torch Variables
            agent_actions = dgn.step(torch_obs, explore=True)
            agent_actions_transposed = np.expand_dims(np.asarray(agent_actions).T, axis=-1)
            # convert actions to numpy arrays

            next_obs, rewards, dones, infos = env.step(agent_actions)
            replay_buffer.push(obs, agent_actions_transposed, rewards, next_obs, dones)
            obs = next_obs
            t += 1
            if t % config.steps_per_update == 0:
                if USE_CUDA:
                    dgn.prep_training(device='gpu')
                else:
                    dgn.prep_training(device='cpu')
                sample = replay_buffer.pop_all(to_gpu=USE_CUDA)
                dgn.update(sample, logger=logger)
                dgn.update_all_targets()
                dgn.prep_rollouts(device='cpu')

        if (ep_i + 1) % config.saving_frequency == 0:
            save_id = 'params_%i.pt' % (ep_i + 1)
            dgn.save(model_dir / curr_run / save_id)
            marl_env_train = make_parallel_env_wolf(
                config.env_id, config.n_rollout_threads,
                config.seed,
                config.close_penalty,
                config.num_players_train
            )
            marl_env_test = make_parallel_env_wolf(
                config.env_id, config.n_rollout_threads,
                config.seed,
                config.close_penalty,
                config.num_players_test
            )
            # Train env
            avgs = []
            dgn.prep_rollouts(device='cpu')
            num_dones, per_worker_rew = [0] * config.n_rollout_threads, [0] * config.n_rollout_threads
            obs = marl_env_train.reset()
            while (all([k < config.eval_eps for k in num_dones])):
                torch_obs = Variable(torch.tensor(obs),
                                     requires_grad=False).float()
                # get actions as torch Variables
                agent_actions = dgn.step(torch_obs)

                # Env step
                next_obs, rewards, dones, infos = marl_env_train.step(agent_actions)
                average_step_rews = compute_average_reward(obs, rewards)
                dones = [any(x) for x in dones]

                per_worker_rew = [k + l for k, l in zip(per_worker_rew, average_step_rews)]
                obs = next_obs

                for idx, flag in enumerate(dones):
                    if flag:
                        if num_dones[idx] < config.eval_eps:
                            num_dones[idx] += 1
                            avgs.append(per_worker_rew[idx])
                        per_worker_rew[idx] = 0

            print("Finished marl train with rewards " + str((sum(avgs) + 0.0) / len(avgs)))
            marl_env_train.close()
            logger.add_scalar('Rewards/marl_train_set', (sum(avgs) + 0.0) / len(avgs), ep_i + 1)

            # Test env
            avgs = []
            dgn.prep_rollouts(device='cpu')
            num_dones, per_worker_rew = [0] * config.n_rollout_threads, [0] * config.n_rollout_threads
            obs = marl_env_test.reset()
            while (all([k < config.eval_eps for k in num_dones])):
                torch_obs = Variable(torch.tensor(obs),
                                     requires_grad=False).float()
                # get actions as torch Variables
                agent_actions = dgn.step(torch_obs)

                # Env step
                next_obs, rewards, dones, infos = marl_env_test.step(agent_actions)
                average_step_rews = compute_average_reward(obs, rewards)
                dones = [any(x) for x in dones]

                per_worker_rew = [k + l for k, l in zip(per_worker_rew, average_step_rews)]
                obs = next_obs

                for idx, flag in enumerate(dones):
                    if flag:
                        if num_dones[idx] < config.eval_eps:
                            num_dones[idx] += 1
                            avgs.append(per_worker_rew[idx])
                        per_worker_rew[idx] = 0

            print("Finished marl eval with rewards " + str((sum(avgs) + 0.0) / len(avgs)))
            marl_env_test.close()
            logger.add_scalar('Rewards/marl_eval_set', (sum(avgs) + 0.0) / len(avgs), ep_i + 1)

            env_train_adhoc = AsyncVectorEnv(
                [
                    make_env_adhoc_wolf('wolfpack-v5', i, config.num_players_train, 2000, config.close_penalty)
                    for i in range(config.n_rollout_threads)
                ]
            )

            env_eval_adhoc = AsyncVectorEnv(
                [
                    make_env_adhoc_wolf(
                        'wolfpack-v5', i, config.num_players_train, 2000, config.close_penalty,
                        implicit_max_player_num=config.num_players_test
                    )
                    for i in range(config.n_rollout_threads)
                ]
            )

            avgs = []
            dgn.prep_rollouts(device='cpu')
            num_dones, per_worker_rew = [0] * config.n_rollout_threads, [0] * config.n_rollout_threads
            obs = env_train_adhoc.reset()
            while (all([k < config.eval_eps for k in num_dones])):
                torch_obs = obs_adhoc_postprocess_wolf(obs['teammate_location'], obs['opponent_info'])
                # Agent computes actions
                agent_actions = dgn.indiv_step(torch_obs)
                # Env step
                next_obs, rewards, dones, infos = env_train_adhoc.step(agent_actions)
                per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
                obs = next_obs
                for idx, flag in enumerate(dones):
                    if flag:
                        if num_dones[idx] < config.eval_eps:
                            num_dones[idx] += 1
                            avgs.append(per_worker_rew[idx])
                        per_worker_rew[idx] = 0

            print("Finished ad hoc train with rewards " + str((sum(avgs) + 0.0) / len(avgs)))
            env_train_adhoc.close()
            logger.add_scalar('Rewards/adhoc_train', (sum(avgs) + 0.0) / len(avgs), ep_i + 1)

            avgs = []
            dgn.prep_rollouts(device='cpu')
            num_dones, per_worker_rew = [0] * config.n_rollout_threads, [0] * config.n_rollout_threads
            obs = env_eval_adhoc.reset()
            while (all([k < config.eval_eps for k in num_dones])):
                torch_obs = obs_adhoc_postprocess_wolf(obs['teammate_location'], obs['opponent_info'])
                # Agent computes actions
                agent_actions = dgn.indiv_step(torch_obs)
                # Env step
                next_obs, rewards, dones, infos = env_eval_adhoc.step(agent_actions)
                per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
                obs = next_obs
                for idx, flag in enumerate(dones):
                    if flag:
                        if num_dones[idx] < config.eval_eps:
                            num_dones[idx] += 1
                            avgs.append(per_worker_rew[idx])
                        per_worker_rew[idx] = 0

            print("Finished ad hoc test with rewards " + str((sum(avgs) + 0.0) / len(avgs)))
            env_eval_adhoc.close()
            logger.add_scalar('Rewards/adhoc_test', (sum(avgs) + 0.0) / len(avgs), ep_i + 1)
            

    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=0, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=16, type=int)
    parser.add_argument("--n_training_threads", default=1, type=int)
    parser.add_argument("--buffer_length", default=64, type=int)
    parser.add_argument("--max_num_steps", default=400000, type=int)
    parser.add_argument("--eps_length", default=200, type=int)
    parser.add_argument("--steps_per_update", default=4, type=int)
    parser.add_argument("--eval_eps", default=5, type=int)
    parser.add_argument("--saving_frequency", default=50, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=1500, type=int)
    parser.add_argument("--init_noise_scale", default=1.0, type=float)
    parser.add_argument("--final_noise_scale", default=0.05, type=float)
    #parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr_crit", default=0.001, type=float)
    parser.add_argument("--close_penalty", default=0.5, type=float)
    parser.add_argument("--lr_pol", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--temp", default=0.001, type=float)
    parser.add_argument("--reg", default=0.03, type=float)
    parser.add_argument("--alg",
                        default="DGN", type=str,
                        choices=['DGN'])
    parser.add_argument("--discrete_action",
                        action='store_true')
    parser.add_argument('--num_players_train', type=int, default=3, help="maximum num players in train")
    parser.add_argument('--num_players_test', type=int, default=5, help="maximum num players in test")

    config = parser.parse_args()

    run(config)
