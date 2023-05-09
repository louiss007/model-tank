"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-11-20 下午12:18
# @FileName: data_preprocessing.py
# @Email   : quant_master2000@163.com
======================
"""
import pandas as pd
import time
from tqdm import tqdm


def data_concat(raw_data_file, ad_data_file, user_data_file, output_file):
    df_raw = pd.read_csv(raw_data_file)
    df_ad = pd.read_csv(ad_data_file)
    df_user = pd.read_csv(user_data_file)

    df_raw_user = pd.merge(df_raw, df_user, how='left', left_on='user', right_on='userid')
    df_raw_user_ad = pd.merge(df_raw_user, df_ad, how='left', on='adgroup_id')
    df_raw_user_ad.drop(columns=['userid'], inplace=True)
    # print(df_raw_user_ad.columns.values)
    click_hour = ['0']*len(df_raw_user_ad)
    df_raw_user_ad['click_hour'] = click_hour
    for i in tqdm(range(len(df_raw_user_ad)), desc='Processing'):
        df_raw_user_ad['click_hour'][i] = time.strftime('%Y%m%d%H%M%S', time.localtime(df_raw_user_ad['time_stamp'][i]))
    df_raw_user_ad.to_csv(output_file, index=False)


def get_train_and_test_data(data_file, output_train_file, output_test_file):
    df = pd.read_csv(data_file)
    df.rename(columns={'clk': 'label'}, inplace=True)
    df.fillna(-1, inplace=True)
    df_s = df.sort_values(by="click_hour", ascending=True)
    cols = df_s.columns.tolist()
    cols.remove('pid')
    cols.remove('price')
    df_s[['click_hour']] = df_s[['click_hour']].astype('int64')
    df_s[['click_hour']] = df_s[['click_hour']].astype('str')
    df_s[['click_hour']] = df_s[['click_hour']].applymap(lambda x: x[8:10])
    # print(df_s.dtypes)
    df_s.reset_index(inplace=True, drop=True)
    train_samples = int(len(df)*0.85)
    df_train = df_s.iloc[:train_samples]

    df_train.reset_index(inplace=True, drop=True)
    df_test = df_s.iloc[train_samples:]
    df_test.reset_index(inplace=True, drop=True)
    df_train[cols] = df_train[cols].astype('int64')
    df_test[cols] = df_test[cols].astype('int64')
    df_train.to_csv(output_train_file, index=False)
    df_test.to_csv(output_test_file, index=False)


def feat_stat_info(csv_file):
    df = pd.read_csv(csv_file)
    columns = df.columns.tolist()
    # print(columns)
    feat_value2cnt = {}
    for col in columns:
        # print(len(set(df[[col]].values.flatten().tolist())))
        feat_value_cnt = len(set(df[[col]].values.flatten().tolist()))
        print(col, feat_value_cnt)
        if feat_value_cnt < 100:
            print(set(df[[col]].values.flatten().tolist()))
        feat_value2cnt.setdefault(col, [feat_value_cnt])
    new_df = pd.DataFrame(feat_value2cnt, columns=columns)
    print(new_df)


def get_field_size(feat_index_file):
    """

    :param feat_index_file: feat fields encoding file for libsvm, that is a feature map file
    :return:
    """
    feature_fields = dict()
    with open(feat_index_file) as fin:
        for line in fin.readlines():
            lines = line.strip().split(':')
            if len(lines) < 2:
                continue
            feat_ind, feat_val = lines
            feature_fields.setdefault(feat_ind, set())
            feature_fields[feat_ind].add(feat_val)
    field2feat_val_num = dict()
    for k in feature_fields:
        field2feat_val_num.setdefault(k, len(feature_fields.get(k)))

    print(field2feat_val_num)
    print(len(field2feat_val_num))


def trans_csv_file(in_csv_file, out_csv_file):
    df = pd.read_csv(in_csv_file)
    # print(df.columns.values)
    df.rename(columns={'new_user_class_level ': 'new_user_class_level'}, inplace=True)
    df.to_csv(out_csv_file, index=None)


if __name__ == '__main__':
    # data_path = '/home/louiss007/MyWorkShop/dataset/alibaba/tianchi'
    # raw_data_file = '{}/raw_sample_tiny_10w.csv'.format(data_path)
    # ad_data_file = '{}/ad_feature.csv'.format(data_path)
    # user_data_file = '{}/user_profile.csv'.format(data_path)
    # output_file = '../data/rs/ali_tianchi/data_tiny_10w.csv'
    output_train_file = '../data/rs/ali_tianchi/test_tiny_10w.csv'
    # output_test_file = '../data/rs/ali_tianchi/test_tiny_10w.csv'
    # # data_concat(raw_data_file, ad_data_file, user_data_file, output_file)
    # # get_train_and_test_data(output_file, output_train_file, output_test_file)
    # feat_stat_info(output_train_file)
    rev_output_train_file = '../data/rs/ali_tianchi/test_tiny_10w.csv'
    trans_csv_file(output_train_file, rev_output_train_file)

    # lib_svm_train_file = '/home/louiss007/MyWorkShop/dataset/ipinyou/make-ipinyou-data/2997/featindex.txt'
    # get_field_size(lib_svm_train_file)
