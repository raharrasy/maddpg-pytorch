#!/bin/bash

source activate pytorch_p36
python main.py 'Foraging-8x8-3f-v0' parameters --lr_crit=0.00001 --lr_pol=0.00001  --reg=0.01 --seed=200
