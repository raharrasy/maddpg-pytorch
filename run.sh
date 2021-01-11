#!/bin/bash

source activate pytorch_p36
python main.py 'Foraging-8x8-3f-v0' parameters --lr_crit=0.0001 --lr_pol=0.000001  --reg=0.001 --seed=200
