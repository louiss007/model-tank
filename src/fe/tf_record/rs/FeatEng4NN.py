"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-10-7 下午10:31
# @FileName: FeatEng4NN.py
# @Email   : quant_master2000@163.com
======================
"""
import functools
import tensorflow as tf


class FeatEng4NN:
    def __init__(self):
        pass

    def data_preprocessing_for_categorical(self, categorical_features):
        categorical_columns = []
        for feature, vocab in categorical_features.items():
            cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
                key=feature, vocabulary_list=vocab)
            # categorical_columns.append(tf.feature_column.indicator_column(cat_col))
            categorical_columns.append(cat_col)
        return categorical_columns

    def data_preprocessing_for_categorical_v2(self, categorical_features):
        categorical_columns = self.data_preprocessing_for_categorical(categorical_features)
        feat_value_cnt = 0
        for feat in categorical_features:
            feat_value_cnt += len(categorical_features.get(feat))

        print('==============feat_value_cnt:{0}=============='.format(feat_value_cnt))
        categorical_columns_embed = []
        for feat_col in categorical_columns:
            cat_col_embed = tf.feature_column.embedding_column(feat_col, dimension=feat_value_cnt)
            categorical_columns_embed.append(cat_col_embed)
            # categorical_columns_embed.append(tf.feature_column.indicator_column(cat_col))
        return categorical_columns_embed

    def process_continuous_data(self, mean, data):
        """
        标准化数据
        :param mean:
        :param data:
        :return:
        """
        data = tf.cast(data, tf.float32) * 1/(2*mean)
        return tf.reshape(data, [-1, 1])

    def data_preprocessing_for_numerical(self, numerical_features):
        numerical_columns = []
        for feature in numerical_features:
            num_col = tf.feature_column.numeric_column(
                feature, normalizer_fn=functools.partial(
                    self.process_continuous_data,
                    numerical_features[feature]
                )
            )
            numerical_columns.append(num_col)
        return numerical_columns

    def build_categorical_feature(self, values):
        vals = [val.encode() for val in values]
        f = tf.train.Feature(bytes_list=tf.train.BytesList(value=vals))
        return f

    def build_numerical_feature(self, value):
        f = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        return f

    def build_categorical_features(self, categorical_features):
        features = {}
        for key in categorical_features:
            values = categorical_features.get(key)
            f = self.build_categorical_feature(values)
            features.setdefault(key, f)
        return features

    def build_numerical_features(self, numerical_features):
        features = {}
        for key in numerical_features:
            value = numerical_features.get(key)
            f = self.build_numerical_feature(value)
            features.setdefault(key, f)
        return features


if __name__ == '__main__':
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

    feat_eng = FeatEng4NN()
    cate_features = feat_eng.build_categorical_features(categorical_features)
    nume_features = feat_eng.build_numerical_features(numerical_features)
    features = {}
    features.update(cate_features)
    features.update(nume_features)
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    columns = []
    categorical_columns = feat_eng.data_preprocessing_for_categorical_v2(categorical_features)
    numerical_columns = feat_eng.data_preprocessing_for_numerical(numerical_features)
    columns.extend(categorical_columns)
    columns.extend(numerical_columns)
    # tf.io.parse_example([], ), 第一个参数要带中括号！！！
    features = tf.io.parse_example([tf_example.SerializeToString()], tf.feature_column.make_parse_example_spec(columns))
    inputs = tf.feature_column.input_layer(features=features, feature_columns=columns)
    print('=======================shape:{}======================'.format(inputs.get_shape()[1]))

    sess = tf.Session()
    variables_initner = tf.global_variables_initializer()
    tables_initner = tf.tables_initializer()
    sess.run(variables_initner)
    sess.run(tables_initner)
    v = sess.run(inputs)
    sess.close()
    print(v)