nohup python -u main.py --dataset=$1 --model=$2 --drop_rate=$3 --num_gradual=$4 --gpu=$5 --eval_freq=2000 --batch_size=1024 > log/$1/$2_$3-$4.log 2>&1 &
