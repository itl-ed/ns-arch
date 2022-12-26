#!/bin/bash
for ((i=$1;i<=$2;i++))
do
    python tools/exp1/run.py seed=$i +agent.model_path=$PWD/assets/agent_models/injected_42.ckpt exp1.strat_feedback=minHelp
    python tools/exp1/run.py seed=$i +agent.model_path=$PWD/assets/agent_models/injected_42.ckpt exp1.strat_feedback=medHelp
    python tools/exp1/run.py seed=$i +agent.model_path=$PWD/assets/agent_models/injected_42.ckpt exp1.strat_feedback=maxHelp
    python tools/exp1/run.py seed=$i +agent.model_path=$PWD/assets/agent_models/injected_42.ckpt exp1.strat_feedback=maxHelp agent.strat_generic=semNeg
    python tools/exp1/run.py seed=$i +agent.model_path=$PWD/assets/agent_models/injected_42.ckpt exp1.strat_feedback=maxHelp agent.strat_generic=semNegScal
done
