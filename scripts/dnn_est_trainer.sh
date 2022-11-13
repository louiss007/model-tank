#!/usr/bin/env bash

cur_path=$(cd $(dirname $0);pwd)
echo ${cur_path}
pre_path=${cur_path%/*}
echo ${pre_path}

export PYTHONPATH=${pre_path}

epoch=100
if [ $1 == 'fnn' ]
then
    model_name=fnn
    layers=784,128,256,32
elif [ $1 == 'cnn' ]
then
    model_name=cnn
    layers=784,128,256,32
else
    echo 'model $1 not support!'
    exit
fi
task_type=classification
input_train_file=${pre_path}/data/cv/mnist
output_path=${pre_path}/output/tf_estimator/${model_name}
batch_size=128
dropout=0.8
lr=0.001
input_size=784
display_step=200


echo 'model '${model_name}' is training with estimator...'

python3 ${pre_path}/src/models/dl/dnn_est_trainer.py \
    --task-type ${task_type} \
    --model-name ${model_name} \
    --input ${input_train_file} \
    --output ${output_path} \
    --epochs ${epoch} \
    --batch-size ${batch_size} \
    --dropout ${dropout} \
    --lr ${lr} \
    --input-size ${input_size} \
    --layers ${layers} \
    --display-step ${display_step}