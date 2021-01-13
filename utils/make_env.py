"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

def make_env(scenario_name, benchmark=False, discrete_action=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:        
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.benchmark_data,
                            discrete_action=discrete_action)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation,
                            discrete_action=discrete_action)
    return env

def make_env_lbf(env_id, seed=1285, effective_max_num_players=3, with_shuffle=True, multi_agent=True):
    import gym
    import lbforaging
    def _init():
        env = gym.make(
            env_id, seed=seed,
            effective_max_num_players=effective_max_num_players,
            init_num_players=effective_max_num_players,
            with_shuffle=with_shuffle,
            multi_agent=multi_agent
        )
        return env

    return _init

def make_env_wolf(env_id, seed=1285, close_penalty=0.5, implicit_max_player_num=3, max_player_num=5):
    import gym
    import Wolfpack_gym
    def _init():
        env = gym.make(
            env_id, seed=seed,
            num_players=implicit_max_player_num,
            close_penalty=close_penalty,
            implicit_max_player_num=implicit_max_player_num,
            max_player_num=max_player_num
        )
        return env

    return _init