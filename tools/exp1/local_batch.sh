for ((i=$1;i<=$2;i++))
    do
        echo "min_${i}"
        python tools/exp1/run.py -x1tf min -x1df easy -x1ne 25 -x1rs ${i} -lp output/injected.ckpt > output/exp1_res/log_easy_min_zeroInit_${i}.txt
        echo "med_${i}"
        python tools/exp1/run.py -x1tf med -x1df easy -x1ne 25 -x1rs ${i} -lp output/injected.ckpt > output/exp1_res/log_easy_med_zeroInit_${i}.txt
        echo "max_${i}"
        python tools/exp1/run.py -x1tf max -x1df easy -x1ne 25 -x1rs ${i} -lp output/injected.ckpt > output/exp1_res/log_easy_max_zeroInit_${i}.txt
    done