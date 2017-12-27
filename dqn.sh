#!/bin/bash
source activate dqn
python dqn.py $@ --jobid=$SLURM_JOB_ID

    
