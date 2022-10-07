"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-10-7 下午10:33
# @FileName: csv_to_tfrecord.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
import pandas as pd
from FeatEng4NN import FeatEng4NN
import json
import time


def load_csv_data(csv_file):
    df = pd.read_csv(csv_file)
    df.fillna(value=0, inplace=True)
    return df


def build_numerical_feature(value):
    f = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    return f


def csv_to_tfrecord_numerical(csv_file, tfrecord_file):
    tfrecord_file = '../../data/ubiquant-market-prediction/train_01.tfrecord'
    writer = tf.io.TFRecordWriter(tfrecord_file)
    p_feature = {}
    chunk_size = 200000
    feature = dict()
    reader = pd.read_csv(csv_file, iterator=True)
    sample_cnt = 0
    loop = False
    tmp_df = reader.get_chunk(chunk_size)
    cols = tmp_df.filter(like="f_").columns.tolist()
    cols.append('target')
    print('cols:{0}'.format(cols))
    for col in cols:
        p_feature.setdefault(col, tf.io.FixedLenFeature([], tf.float32))
    real_size = len(tmp_df)
    for i in range(real_size):
        for col in cols:
            feature[col] = build_numerical_feature(tmp_df[col][i])
        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
    writer.close()
    sample_cnt += real_size
    print('=============sample_cnt:{0} done!================'.format(sample_cnt))
    if sample_cnt == chunk_size:
        loop = True
    i = 2
    while loop:
        try:
            tfrecord_file = '../../data/ubiquant-market-prediction/train_%02d.tfrecord' % i
            writer = tf.io.TFRecordWriter(tfrecord_file)
            tmp_df = reader.get_chunk(chunk_size)
            real_size = len(tmp_df)
            tmp_df.reset_index(drop=True, inplace=True)
            for k in range(len(tmp_df)):
                # feature = dict()
                for col in cols:
                    feature[col] = build_numerical_feature(tmp_df[col][k])
                features = tf.train.Features(feature=feature)
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
            writer.close()
            i += 1
            sample_cnt += real_size
            print('=============sample_cnt:{0} done!================'.format(sample_cnt))
        except StopIteration:
            loop = False
            print("read csv file finished!")
            break
    writer.close()
    print('=================sample_cnt:{0}===================='.format(sample_cnt))
    # cols = tmp_df.filter(like="f_").columns.tolist()
    # fi = open(feat_json_file, 'r')
    # var2mean = json.loads(fi.read())
    # feat_eng = FeatEng4NN()
    # features = feat_eng.build_numerical_features(var2mean)
    # tf_example = tf.train.Example(features=tf.train.Features(feature=features))


def csv_to_tfrecord(csv_file, tfrecord_file):
    categorical_features = {
        'sex': ['male', 'female'],
        'class': ['First', 'Second', 'Third'],
        'deck': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        'embark_town': ['Cherbourg', 'Southhampton', 'Queenstown'],
        'alone': ['y', 'n']
    }

    numerical_features = {
        'age': 29.631308,
        'n_siblings_spouses': 0.545455,
        'parch': 0.379585,
        'fare': 34.385399
    }

    tf_cols = ['age', 'parch', 'n_siblings_spouses', 'fare', 'survived']
    data = load_csv_data(csv_file)
    cols = data.columns.tolist()
    print(cols)
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    feat_eng = FeatEng4NN()
    parse_feature = {}
    for i in range(len(data)):
        feature = dict()
        for col in tf_cols:
            if col in ['age', 'fare']:
                feature.setdefault(col, feat_eng.build_numerical_feature(data[col][i]))
                parse_feature.setdefault(col, tf.FixedLenFeature([], tf.float32))
            else:
                feature.setdefault(col, tf.train.Feature(int64_list=tf.train.Int64List(value=[data[col][i]])))
                parse_feature.setdefault(col, tf.FixedLenFeature([], tf.int64))
            # if col in numerical_features:
            #     feature.setdefault(col, feat_eng.build_numerical_feature(data[col][i]))
            #     parse_feature.setdefault(col, tf.FixedLenFeature([], tf.float32))
            # elif col in categorical_features:
            #     feature.setdefault(col, feat_eng.build_categorical_feature([data[col][i]]))
            #     parse_feature.setdefault(col, tf.FixedLenFeature([], tf.int64))
            # else:
            #     feature.setdefault(col, tf.train.Feature(int64_list=tf.train.Int64List(value=[data[col][i]])))
            #     parse_feature.setdefault(col, tf.FixedLenFeature([], tf.int64))

        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
    writer.close()
    print('csv to tfrecord finished!')
    return parse_feature, cols


def display_tfrecord(tfrecord_file):
    item = next(tf.python_io.tf_record_iterator(tfrecord_file))
    print(tf.train.Example.FromString(item))


def load_tfrecord(tfrecord_files, p_feature):
    tfrecord_file_queue = tf.train.string_input_producer(tfrecord_files)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tfrecord_file_queue)
    features = tf.parse_single_example(serialized_example, features=p_feature)

    age = features['age']
    # sex = features['sex']
    # pclass = features['class']
    parch = features['parch']
    sibsp = features['n_siblings_spouses']
    fare = features['fare']
    label = features['survived']
    age, parch, sibsp, fare, label = tf.train.batch([age, parch, sibsp, fare, label],
                       batch_size=32, capacity=500)
    print(age.shape)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        age_ = sess.run(age)
        print(age_)
        coord.request_stop()
        coord.join(threads)
    # with tf.Session() as sess:
    # sess = tf.Session()
    # init_global = tf.global_variables_initializer()
    # sess.run(init_global)
    # # # 第四步
    # # coord = tf.train.Coordinator()
    # # # 第五步：启动队列
    # # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # '''第六步，这里面就可以查看数据，将数据“喂“给网络了 '''
    # age_ = sess.run(age)
    # print(age_)
    #
    # # # 第七步
    # # coord.request_stop()
    # # coord.join(threads=threads)
    # print('完结！')


if __name__ == '__main__':
    # csv_file = '../../data/titanic/train.csv'
    # tfrecord_file = '../../data/titanic/train.tfrecord'
    # tfrecord_files = [tfrecord_file]
    # p_feature, cols = csv_to_tfrecord(csv_file, tfrecord_file)
    # # display_tfrecord(tfrecord_file)
    # load_tfrecord(tfrecord_files, p_feature)

    csv_file = '../../data/ubiquant-market-prediction/train.csv'
    tfrecord_file = '../../data/ubiquant-market-prediction/train.tfrecord'
    start = time.time()
    csv_to_tfrecord_numerical(csv_file, tfrecord_file)
    end = time.time()
    print('==========time:{0}============'.format(end-start))
    # display_tfrecord(tfrecord_file)
