#!/bin/bash

source activate pytorch_p36
python main_dgn.py 'wolfpack-v5' parameters --lr_crit=0.0001 --reg=0.03 --seed=700


