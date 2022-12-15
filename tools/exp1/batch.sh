#!/bin/bash
for ((i=$1;i<=$2;i++))
do
    python tools/exp1/run.py exp1.difficulty=fineEasy exp1.strat_feedback=minHelp +agent.model_path=$PWD/assets/agent_models/injected_42.ckpt seed=$i
    python tools/exp1/run.py exp1.difficulty=fineEasy exp1.strat_feedback=medHelp +agent.model_path=$PWD/assets/agent_models/injected_42.ckpt seed=$i
    python tools/exp1/run.py exp1.difficulty=fineEasy exp1.strat_feedback=maxHelp +agent.model_path=$PWD/assets/agent_models/injected_42.ckpt seed=$i
done
