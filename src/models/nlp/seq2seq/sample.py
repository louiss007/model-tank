"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 23-1-23 下午11:05
# @FileName: sample.py
# @Email   : quant_master2000@163.com
======================
"""
import pickle
import os


class Sample(object):

    def __init__(self, src_file, tgt_file):
        self.special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        self.src_lines = self.load_data(src_file)
        self.tgt_lines = self.load_data(tgt_file)
        self.src_max_len = max([len(line) for line in self.src_lines])
        self.tgt_max_len = max([len(line) for line in self.tgt_lines])

        self.src_seq_list, self.tgt_seq_list = self.make_input_seq_list()
        self.src_id2token, self.src_token2id, self.src_token_list = self.build_token_vocab(self.src_seq_list)
        self.src_token_size = len(self.src_token_list)

        self.tgt_id2token, self.tgt_token2id, self.tgt_token_list = self.build_token_vocab(self.tgt_seq_list)
        self.tgt_token_size = len(self.tgt_token_list)

    @staticmethod
    def load_data(in_file):
        with open(in_file, 'r') as fi:
            lines = [line.strip() for line in fi.readlines()]
        return lines

    def make_input_seq_list(self):
        src_seq_list = self.make_seq_list(self.src_lines)
        tgt_seq_list = self.make_seq_list(self.tgt_lines)
        return src_seq_list, tgt_seq_list

    @staticmethod
    def make_seq_list(lines, split_char=None):
        lines = [line.lower() for line in lines]
        if split_char is None:
            seq_list = [[token for token in seq] for seq in lines]
        else:
            seq_list = [[token for token in seq.split(split_char)] for seq in lines]
        return seq_list

    def build_token_vocab(self, seq_list):
        """ 构造token映射表 """
        token_list = list(set([token for seq in seq_list for token in seq]))
        # 这里要把四个特殊字符添加进词典
        id2token = {idx: word for idx, word in enumerate(self.special_tokens + token_list)}
        token2id = {word: idx for idx, word in id2token.items()}
        return id2token, token2id, token_list

    def trans_seq_list_2_ids(self):
        src_id_seq_list, src_id_seq_len_list = self.seq_list_2_ids(self.src_seq_list, self.src_token2id, self.src_max_len)
        tgt_id_seq_list, tgt_id_seq_len_list = self.seq_list_2_ids(self.tgt_seq_list, self.tgt_token2id, self.tgt_max_len)
        return src_id_seq_list, src_id_seq_len_list, tgt_id_seq_list, tgt_id_seq_len_list

    @staticmethod
    def seq_list_2_ids(seq_list, token2id, max_len):
        id_seq_list = []
        id_seq_len_list = []
        for seq in seq_list:
            id_seq = []
            for token in seq:
                id_seq.append(token2id.get(token, '<UNK>'))
            id_seq_list.append(id_seq)
            seq_len = len(seq)
            id_seq_len_list.append(seq_len)
        id_seq_list_pad = [id_seq + [token2id['<PAD>']] * (max_len-len(id_seq)) for id_seq in id_seq_list]
        return id_seq_list_pad, id_seq_len_list

    def save_pkl(self, out_path):
        src_id_seq_list, src_id_seq_len_list, tgt_id_seq_list, tgt_id_seq_len_list = self.trans_seq_list_2_ids()
        fo1 = open(os.path.join(out_path, 'src_id_seq_list.pkl'), 'wb')
        fo2 = open(os.path.join(out_path, 'src_id_seq_len_list.pkl'), 'wb')
        fo3 = open(os.path.join(out_path, 'tgt_id_seq_list.pkl'), 'wb')
        fo4 = open(os.path.join(out_path, 'tgt_id_seq_len_list.pkl'), 'wb')
        pickle.dump(src_id_seq_list, fo1)
        pickle.dump(src_id_seq_len_list, fo2)
        pickle.dump(tgt_id_seq_list, fo3)
        pickle.dump(tgt_id_seq_len_list, fo4)
        fo1.close()
        fo2.close()
        fo3.close()
        fo4.close()

    def load_pkl(self, in_path):
        if not os.path.exists(os.path.join(in_path, 'src_id_seq_list.pkl')):
            print("no exist sample, processing data...")
            self.save_pkl(in_path)
        else:
            fi1 = open(os.path.join(in_path, 'src_id_seq_list.pkl'), 'rb')
            fi2 = open(os.path.join(in_path, 'src_id_seq_len_list.pkl'), 'rb')
            fi3 = open(os.path.join(in_path, 'tgt_id_seq_list.pkl'), 'rb')
            fi4 = open(os.path.join(in_path, 'tgt_id_seq_len_list.pkl'), 'rb')
            src_id_seq_list = pickle.load(fi1)
            src_id_seq_len_list = pickle.load(fi2)
            tgt_id_seq_list = pickle.load(fi3)
            tgt_id_seq_len_list = pickle.load(fi4)
            return src_id_seq_list, src_id_seq_len_list, tgt_id_seq_list, tgt_id_seq_len_list


if __name__ == '__main__':
    src_file = 'data/nlp/seq2seq/src.txt'
    tgt_file = 'data/nlp/seq2seq/tgt.txt'
    sample = Sample(src_file, tgt_file)
    # sample.save_pkl('data/nlp/seq2seq')
    src_id_seq_list, src_id_seq_len_list, tgt_id_seq_list, tgt_id_seq_len_list = sample.load_pkl('data/nlp/seq2seq')
    print(src_id_seq_list)
