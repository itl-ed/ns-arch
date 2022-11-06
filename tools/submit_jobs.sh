#!/bin/bash
# $1 slurm node
# $2 experiment txt
# $3 line number begin
# $4 line number end
for ((i=$3;i<=$4;i++))
do
        COMM="sbatch --array=$i-$i%1 --time=1-06:00:00 --gres=gpu:1 --mem=12000 --cpus-per-task=6 --nodelist=$1 --parsable tools/slurm_arrayjob.sh $2"
        echo $COMM
        RES=$(eval $COMM)
done
