#!/bin/bash
declare -a a0_exps=(-4 -5 -6)
declare -a eps_exps=(-8 -7 -6 -5 -4 -3)

for eps_e in ${eps_exps[@]}
do
    for a0_e in ${a0_exps[@]}
    do
        a0eps_e=$(( $a0_e - $eps_e ))
        a0eps="1e$a0eps_e"
        eps="1e$eps_e"
        COMM="python tools/vision/train.py agent.model=na vision=train vision/task=fs_classify_cls seed=42 vision.optim.init_lr_over_eps=$a0eps vision.optim.eps=$eps vision.optim.max_steps=150"
        eval $COMM
    done
done
