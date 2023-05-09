"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-12-4 下午6:08
# @FileName: input_layer_test.py
# @Email   : quant_master2000@163.com
======================
"""

import tensorflow as tf


def input_layer_one_hot():
    features = {
        'sales': [[5], [10], [8], [9]],
        'department': ['sports', 'sports', 'gardening', 'gardening']
    }

    department_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['sports', 'gardening']
    )
    department_column = tf.feature_column.indicator_column(department_column)
    columns = [
        tf.feature_column.numeric_column('sales'),
        department_column
    ]

    inputs = tf.feature_column.input_layer(features, columns)

    var_init = tf.global_variables_initializer()
    table_init = tf.tables_initializer()

    sess = tf.Session()
    sess.run((var_init, table_init))
    res = sess.run(inputs)
    print(res)


def input_layer_embedding():
    features = {'aa': [[2], [1], [3]]}
    # 特征列 feature_column
    aa_fc = tf.compat.v1.feature_column.categorical_column_with_identity('aa', num_buckets=9, default_value=0)
    print(aa_fc)
    # aa_fc = tf.feature_column.indicator_column(aa_fc)
    '''对于维度特别大的feature_column, 使用 embedding_column, 过于稀疏的特征对模型影响比较大 '''
    aa_fc = tf.compat.v1.feature_column.embedding_column(aa_fc, dimension=4)

    # 组合特征列 feature_columns
    columns = [aa_fc]
    # 输入层
    inputs = tf.compat.v1.feature_column.input_layer(features=features, feature_columns=columns)

    variables_init = tf.compat.v1.global_variables_initializer()
    table_init = tf.compat.v1.tables_initializer()
    sess = tf.compat.v1.Session()
    sess.run(variables_init)
    sess.run(table_init)
    val = sess.run(inputs)
    print(val)


def input_layer_cross():
    # 特征数据
    features = {
        'sex': [1, 2, 1, 1, 2],
        'department': ['sport', 'sport', 'drawing', 'gardening', 'travelling'],
        'age': [3, 2, 6, 5, 4]
    }
    # 特征列
    department = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['sport', 'drawing', 'gardening', 'travelling'], dtype=tf.string
    )
    sex = tf.feature_column.categorical_column_with_identity('sex', num_buckets=2, default_value=0)
    sex_department = tf.feature_column.crossed_column([department, sex], 10)
    # sex_department = tf.feature_column.crossed_column([features['department'],features['sex']], 16)
    sex_department = tf.feature_column.indicator_column(sex_department)
    # 组合特征列
    columns = [sex_department, tf.feature_column.indicator_column(department)]

    # 输入层（数据，特征列）
    inputs = tf.feature_column.input_layer(features, columns)

    sess = tf.compat.v1.Session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(tf.compat.v1.tables_initializer())
    sess.run(init)
    # v = sess.run(inputs)
    # from scipy import sparse
    # sv = sparse.csr_matrix(v)
    # for kv in sv:
    #     print(type(kv))
    # for k in sv.indices:
    #     print(k)
    # print(type(v))
    ind, val = dense_to_sparse(inputs)
    ind, val = sess.run([ind, val])
    print(ind)
    print(val)


def cross_items(item_lst):
    import itertools
    cross_items = list(itertools.combinations(item_lst, 2))
    print(cross_items)


def dense_to_sparse(dense_tensor):
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(dense_tensor, zero)
    indices = tf.where(where)
    vals = tf.gather_nd(dense_tensor, indices)
    return indices, vals


if __name__ == '__main__':
    # input_layer_one_hot()
    # input_layer_embedding()
    input_layer_cross()
    # item_lst = [1, 2, 3]
    # cross_items(item_lst)

