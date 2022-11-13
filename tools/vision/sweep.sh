#!/bin/bash
declare -a a0s=("1e-6" "3e-6" "1e-5" "3e-5" "1e-4")
declare -a b1ms=("1e-2" "1e-1" "1e-0")
declare -a b2ms=("1e-4" "1e-3" "1e-2")
declare -a epss=("1e-9" "1e-8" "1e-7" "1e-6")
declare -a Bs=("32" "64" "128" "256" "512")
declare -a Ks=("4" "8" "16")

for a0 in ${a0s[@]}
do
    for eps in ${epss[@]}
    do
        for b1m in ${b1ms[@]}
        do
            for b2m in ${b2ms[@]}
            do
                COMM="python tools/vision/train.py vision=train vision/task=fs_classify_cls seed=15213 vision.optim.init_lr=$a0 vision.optim.beta1_1m=$b1m vision.optim.beta2_1m=$b2m vision.optim.eps=$eps vision.data.batch_size=64 vision.data.num_exs_per_conc=8 vision.optim.max_steps=10000"
                eval $COMM
            done
        done
    done
done
