"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-10-16 下午7:18
# @FileName: text_to_tfrecord_for_sm.py
# @Email   : quant_master2000@163.com
======================
"""
import json
import codecs
import tensorflow as tf
from tqdm import tqdm
import utils.mt_utils as util


"""
transfer text to tfrecord format for sm model for training
"""


def text2id(text, vocab2id, max_size):
    text = [vocab2id[c] for c in text if c in vocab2id]
    text = text[: max_size] + [0] * (max_size - len(text))
    return text


def text_to_tfrecord_for_sm(in_file, out_file, v_path, q_size=60, d_size=200):
    vocab2id = json.load(codecs.open(v_path, 'r', 'utf-8'))
    fw = tf.io.TFRecordWriter(out_file)
    count = 0
    with codecs.open(in_file, "r", "utf-8") as fi:
        for line in tqdm(fi):
            count += 1
            line = line.strip().split("\t")
            query = text2id(line[0], vocab2id, q_size)
            doc = text2id(line[1], vocab2id, d_size)
            feed_dict = {
                "query": util.build_int_feature(query),
                "doc": util.build_int_feature(doc),
                "label": util.build_int_feature([1])
            }
            example = tf.train.Example(features=tf.train.Features(feature=feed_dict))
            serialized = example.SerializeToString()
            fw.write(serialized)
    print("sample cnt: %d" % count)
    fw.close()


def text_to_tfrecord_for_sm_eval(in_file, out_file, v_path, q_size=60, d_size=200):
    vocab2id = json.load(codecs.open(v_path, 'r', 'utf-8'))
    fw = tf.io.TFRecordWriter(out_file)
    count = 0
    with codecs.open(in_file, "r", "utf-8") as fi:
        for line in tqdm(fi):
            count += 1
            line = line.strip().split("\t")
            query = text2id(line[0], vocab2id, q_size)
            doc = text2id(line[1], vocab2id, d_size)
            feed_dict = {
                "query": util.build_int_feature(query),
                "doc": util.build_int_feature(doc),
                "label": util.build_int_feature([1])
            }
            example = tf.train.Example(features=tf.train.Features(feature=feed_dict))
            serialized = example.SerializeToString()
            fw.write(serialized)
            if count > 999:
                break
    print("sample cnt: %d" % count)
    fw.close()


if __name__ == "__main__":
    # text_to_tfrecord_for_sm(
    #     "data/nlp/sm/data.txt",
    #     "data/nlp/sm/train.tfrecord",
    #     "config/nlp/sm/vocab2id.json",
    #     q_size=60,
    #     d_size=200
    # )

    text_to_tfrecord_for_sm_eval(
        "data/nlp/sm/data.txt",
        "data/nlp/sm/test.tfrecord",
        "config/nlp/sm/vocab2id.json",
        q_size=60,
        d_size=200
    )
