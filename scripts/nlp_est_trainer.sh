#!/usr/bin/env bash

cur_path=$(cd $(dirname $0);pwd)
echo ${cur_path}
pre_path=${cur_path%/*}
echo ${pre_path}

export PYTHONPATH=${pre_path}

epoch=3
if [ $1 == 'dssm' ]
then
    model_name=dssm
elif [ $1 == 'clsm' ]
then
    model_name=clsm
else
    model_name=dssm
fi
task_type=classification
input=${pre_path}/data/nlp/sm/train.tfrecord
test_file=${pre_path}/data/nlp/sm/test.tfrecord
output=${pre_path}/output/${model_name}
batch_size=64
dropout=0.8
lr=0.001
input_size=784
layers=128,64
display_step=100
token_vocab_size=18672
query_max_length=60     # dim must be in coincide with train.tfrecord
doc_max_length=200      # dim must be in coincide with train.tfrecord


echo 'model '${model_name}' is training...'

python3 ${pre_path}/src/models/nlp/nlp_sm_trainer_est.py \
    --task_type ${task_type} \
    --model_name ${model_name} \
    --input ${input} \
    --output ${output} \
    --test_file_path ${test_file} \
    --epochs ${epoch} \
    --batch_size ${batch_size} \
    --dropout ${dropout} \
    --lr ${lr} \
    --input_size ${input_size} \
    --layers ${layers} \
    --token_vocab_size ${token_vocab_size} \
    --query_max_length ${query_max_length} \
    --doc_max_length ${doc_max_length} \
    --display_step ${display_step}