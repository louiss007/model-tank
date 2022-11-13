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
model_path=${pre_path}/output/${model_name}
test_file=${pre_path}/data/cv/mnist/test.tfrecord
result_file=${pre_path}/data/cv/mnist/test_result.txt


echo 'model '${model_name}' is inferring...'

python3 ${pre_path}/src/models/dl/dnn_predictor.py \
    --task-type ${task_type} \
    --model-name ${model_name} \
    --model-path ${model_path} \
    --input ${test_file} \
    --output ${result_file}
