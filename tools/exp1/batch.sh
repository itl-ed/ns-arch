#!/bin/bash
declare -a Difficulty=("base" "easy")
declare -a TeacherStrat=("min" "med")
mkdir -p "$3/ns-arch/output"

for diff in ${Difficulty[@]}
do
    for tstrat in ${TeacherStrat[@]}
    do
        for ((i=$1;i<=$2;i++))
        do
            COMM="python tools/run.py -ic -lp wandb://jpstyle/vision_vg_scenegraph/1eg0ctph/checkpoints/epoch\=9-distro.ckpt -x1df $diff -x1tf $tstrat -x1rs $i -op '$3/ns-arch/output' > $3/ns-arch/output/log_${diff}_${tstrat}_zeroInit_${i}.txt"
            eval $COMM
        done
    done
done
