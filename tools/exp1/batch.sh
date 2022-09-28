#!/bin/bash
declare -a Difficulty=("base" "easy")
declare -a TeacherStrat=("min" "med")
mkdir -p "$3/ns-arch/output/exp1_res"

for diff in ${Difficulty[@]}
do
    for tstrat in ${TeacherStrat[@]}
    do
        for ((i=$1;i<=$2;i++))
        do
            COMM="python tools/exp1/run.py -lp assets/models/injected_42.ckpt -x1df $diff -x1tf $tstrat -x1rs $i -x1ne 30 -op '$3/ns-arch/output' > $3/ns-arch/output/exp1_res/log_${diff}_${tstrat}_zeroInit_${i}.txt"
            eval $COMM
        done
    done
done
