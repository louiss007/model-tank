"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 23-1-15 下午8:07
# @FileName: nlp_seq2seq_trainer.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
import pickle
from src.models.nlp.seq2seq.model import Model
from src.models.nlp.seq2seq.seq2seq import Seq2SeqModel
from src.models.nlp.seq2seq.attention_seq2seq import AttentionSeq2SeqModel


class NlpSeq2SeqTrainer(object):

    def __init__(self, task_type):
        self.task_type = task_type

    def train(self, model):
        data_batch = model.make_train_batch(model.input)
        '''
        encoder是否双向
        注意:使用bidirectional，encoder rnn的num_units变为decoder的一半，这是为了能够保证encoder_states和decoder的输入shape能对应上
        '''
        is_encoder_bidirectional = True
        rnn_layer_size = 2
        rnn_num_units = 128
        cell_type = "LSTM"
        lr = 0.001
        decoding_method = "beamSearch"
        # 训练
        model.train(data_batch, layer_size=2, num_units=128, cell_type="LSTM", is_bidirectional=False,
                    decoding_method="greedy", beam_width=10)

    def predict(self):
        dataInfoObj = pickle.load(open("../modelFile/testBasicSeq2Seq/model.dataInfoObj", "rb"))
        model_beam_search = "../modelFile/testBasicSeq2Seq/model_beam_search.ckpt"
        # model_greedy = "../modelFile/testBasicSeq2Seq/model_greedy.ckpt"
        model = Seq2SeqModel(model_path=model_beam_search)
        # 预测
        input = ["abcd", "hello", "word", "kzznhel", "trswatm"]
        source_batch, seq_len = source_seq_list_2_ids(dataInfoObj, input)
        answer_logits = model.predict(source_batch, seq_len)
        print("answer_logits:", answer_logits.shape)
        answer = [[dataInfoObj.target_token_list[index] for index in seq] for seq in answer_logits]
        for i in range(len(input)):
            print(input[i], "  ", "".join(answer[i]))


def main(_):
    """模型训练程序运行入口"""
    # args = tf_arg_parse()
    task_type = args.task_type
    model_name = args.model_name

    if model_name == 'basic':
        model = Model(args, task_type)
    elif model_name == 'seq2seq':
        model = Seq2SeqModel(args, task_type)
    else:
        model = AttentionSeq2SeqModel()

        # print('not support!')
        # return

    if tf.gfile.Exists(model.output):
        tf.gfile.DeleteRecursively(model.output)
    print("==========total train steps=========: ", model.train_steps * model.epochs)
    trainer = NlpSeq2SeqTrainer(task_type)
    trainer.train(model)


if __name__ == '__main__':
    pass
