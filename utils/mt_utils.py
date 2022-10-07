"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-10-7 下午9:55
# @FileName: mt_utils.py
# @Email   : quant_master2000@163.com
======================
"""
import yaml
import codecs
import tensorflow as tf
import os


def shell_cmd(cmd):
    ret = os.system(cmd)
    if not ret:
        print('{0} succeed!'.format(cmd))
        return True
    else:
        print('{0} failed!'.format(cmd))
        return False


def read_yaml(infile):
    fi = codecs.open(infile, 'r', 'utf-8')
    data = fi.read()
    fi.close()
    kv_data = yaml.load(data, yaml.FullLoader)
    return kv_data


def get_dataset_from_csv(csv_file, target_col, batch_size, feat_cols=None):
    if feat_cols is None:
        ds = tf.data.experimental.make_csv_dataset(
            csv_file,
            batch_size=batch_size,
            label_name=target_col,
            na_value="?",
            num_epochs=1,
            ignore_errors=True
        )
    else:
        ds = tf.data.experimental.make_csv_dataset(
          csv_file,
          batch_size=batch_size,
          select_columns=feat_cols,
          label_name=target_col,
          na_value="?",
          num_epochs=1,
          ignore_errors=True
        )
    return ds
