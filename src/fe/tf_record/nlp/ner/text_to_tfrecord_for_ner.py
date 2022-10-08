"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-10-8 下午10:33
# @FileName: text_to_tfrecord_for_ner.py
# @Email   : quant_master2000@163.com
======================
"""
import json
import tokenization
import collections
import tensorflow as tf


def process_one_example(text, label):
    """
    :param text:
    :param label:
    :return:
    """
    text_seq = list(text)
    label_seq = list(label)
    tokens = []
    labels = []
    for i, word in enumerate(text_seq):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = label_seq[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                print("some unknown token...")
                labels.append(labels[0])
    # tokens = tokenizer.tokenize(example.text)  -2 的原因是因为序列需要加一个句首和句尾标志
    if len(tokens) >= max_seq_len - 1:
        tokens = tokens[0:(max_seq_len - 2)]
        labels = labels[0:(max_seq_len - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # [CLS] [SEP] 可以为 他们构建标签，或者 统一到某个标签，反正他们是不变的，基本不参加训练 即：x-l 永远不变
    label_ids.append(0)  # label2id["[CLS]"]
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label2id[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(0)  # label2id["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("**NULL**")
    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len
    assert len(label_ids) == max_seq_len

    feature = (input_ids, input_mask, segment_ids, label_ids)
    return feature


def text_to_tfrecord_for_ner(in_file, tf_record_file):
    """
    生成训练数据， tf.record, 单标签分类模型, 随机打乱数据
    :param in_file:
    :param tf_record_file:
    :return:
    """
    fi = open(in_file, 'r')
    writer = tf.io.TFRecordWriter(tf_record_file)
    example_count = 0

    for line in fi.readlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        text_len = len(data['text'])
        labels = ['O'] * text_len
        for k, v in data['label'].items():
            for kk, vv in v.items():
                for vvv in vv:
                    span = vvv
                    s = span[0]
                    e = span[1] + 1
                    # print(s, e)
                    if e - s == 1:
                        labels[s] = 'S_' + k
                    else:
                        labels[s] = 'B_' + k
                        for i in range(s + 1, e - 1):
                            labels[i] = 'M_' + k
                        labels[e - 1] = 'E_' + k

        feature = process_one_example(list(data['text']), labels)

        features = collections.OrderedDict()
        # 序列标注任务
        features['input_ids'] = build_int_feature(feature[0])
        features['input_mask'] = build_int_feature(feature[1])
        features['segment_ids'] = build_int_feature(feature[2])
        features['label_ids'] = build_int_feature(feature[3])
        if example_count < 5:
            print("*** Example ***")
            print(data['text'])
            print(data['label'])
            print("input_ids: %s" % " ".join([str(x) for x in feature[0]]))
            print("input_mask: %s" % " ".join([str(x) for x in feature[1]]))
            print("segment_ids: %s" % " ".join([str(x) for x in feature[2]]))
            print("label: %s " % " ".join([str(x) for x in feature[3]]))

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
        example_count += 1
        if example_count % 3000 == 0:
            print(example_count)
    print("total example:", example_count)
    writer.close()


def build_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f


if __name__ == '__main__':
    in_trin_file = '../data/train.json'
    train_tf_record = '../data/train.tf_record'
    in_dev_file = '../data/dev.json'
    dev_tf_record = '../data/dev.tf_record'

    vocab_file = 'vocab.txt'
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
    label2id = json.loads(open('label2id.json').read())
    max_seq_len = 64
    text_to_tfrecord_for_ner(in_file=in_trin_file, tf_record_file=train_tf_record)
    text_to_tfrecord_for_ner(in_file=in_dev_file, tf_record_file=dev_tf_record)
