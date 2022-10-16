#!/usr/bin/env bash

cur_path=$(cd $(dirname $0);pwd)
echo ${cur_path}
pre_path=${cur_path%/*}
echo ${pre_path}

export PYTHONPATH=${pre_path}

epoch=3
if [ $1 == 'fnn' ]
then
    model_name=fnn
    layers=784,128,256,32
elif [ $1 == 'cnn' ]
then
    model_name=cnn
    layers=784,128,256,32
elif [ $1 == 'rnn' ]
then
    model_name=rnn
    layers=28,256,32
elif [ $1 == 'lstm' ]
then
    model_name=lstm
    epoch=25
    layers=28,256,32
elif [ $1 == 'bilstm' ]
then
    model_name=bilstm
    epoch=25
    layers=28,256,32
elif [ $1 == 'gru' ]
then
    model_name=gru
    epoch=25
    layers=28,256,32
elif [ $1 == 'gan' ]
then
    model_name=gan
    epoch=5
    layers=784,256,256
elif [ $1 == 'dcgan' ]
then
    model_name=dcgan
    epoch=50
    layers=784,256,256
elif [ $1 == 'alexnet' ]
then
    model_name=alexnet
    layers=784,128,256,32
else
    model_name='fnn'
    layers=784,128,256,32
fi
task_type=classification
input_train_file=${pre_path}/data/cv/mnist/train.tfrecord
output_path=${pre_path}/output/${model_name}
batch_size=128
dropout=0.8
lr=0.001
input_size=784
display_step=200


echo 'model:'${model_name}

python3 ${pre_path}/src/models/dl/dnn_trainer.py \
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