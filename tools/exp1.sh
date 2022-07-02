#!/bin/bash
declare -a Difficulty=("base" "easy")
declare -a TeacherStrat=("min" "med")
mkdir -p output

for diff in ${Difficulty[@]}
do
    for tstrat in ${TeacherStrat[@]}
    do
        for ((i=$1;i<=$2;i++))
        do
            COMM="python tools/exp1.py -ic -lp wandb://jpstyle/vision_vg_scenegraph/1eg0ctph/checkpoints/epoch\=9-distro.ckpt -x1df $diff -x1tf $tstrat -x1rs $i -op 'SCRATCHHOME/ns-arch/output' > SCRATCHHOME/ns-arch/log_${diff}_${tstrat}_zeroInit_${i}.txt"
            eval $COMM
        done
    done
done
