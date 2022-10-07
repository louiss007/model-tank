"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-10-7 下午10:21
# @FileName: get_image2label_id.py
# @Email   : quant_master2000@163.com
======================
"""
import os
import json


def get_image2label_id(in_path, image_class_name2number):
    image2label_id = {}
    sub_path_list = os.listdir(in_path)
    for sub_path in sub_path_list:
        file_path = os.path.join(in_path, sub_path)
        file_list = os.listdir(file_path)
        for image in file_list:
            image2label_id[os.path.join(file_path, image)] = image_class_name2number[sub_path]
    return image2label_id


def save2json(image2label_id, path):
    with open(path, 'w') as fo:
        json.dump(image2label_id, fo)
    print("image2label_id has been saved to json file!")
