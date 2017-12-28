#!/bin/bash
source activate tfgpu
python dqn.py $@ --jobid=$SLURM_JOB_ID

    
