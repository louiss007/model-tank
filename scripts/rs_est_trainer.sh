#!/usr/bin/env bash

cur_path=$(cd $(dirname $0);pwd)
echo ${cur_path}
pre_path=${cur_path%/*}
echo ${pre_path}

export PYTHONPATH=${pre_path}

epoch=3

if [ $1 == 'mlp' ]
then
    model_name=mlp
    layers=128,256,32
elif [ $1 == 'wnd' ]
then
    model_name=wnd
    layers=128,256,32
elif [ $1 == 'deepfm' ]
then
    model_name=deepfm
    layers=128,256,32
else
    model_name=wnd
    layers=128,256,32
fi
task_type=classification
input_train_file=${pre_path}/data/rs/ali_tianchi/train_tiny_10w.csv
input_eval_file=${pre_path}/data/rs/ali_tianchi/test_tiny_10w.csv
#input_train_file='/home/louiss007/MyWorkShop/dataset/ipinyou/make-ipinyou-data/2997/train.yzx.txt'
#input_train_file='/home/louiss007/MyWorkShop/dataset/criteo/tr.libsvm'
output_path=${pre_path}/output/rs/${model_name}
batch_size=32
dropout=0.8
lr=0.0001
display_step=200
nclass=2


echo 'model '${model_name}' is training...'

python3 ${pre_path}/src/models/rs/rs_est_trainer.py \
    --task-type ${task_type} \
    --model-name ${model_name} \
    --input ${input_train_file} \
    --eval-file ${input_eval_file} \
    --output ${output_path} \
    --epochs ${epoch} \
    --batch-size ${batch_size} \
    --dropout ${dropout} \
    --lr ${lr} \
    --layers ${layers} \
    --nclass ${nclass} \
    --display-step ${display_step}