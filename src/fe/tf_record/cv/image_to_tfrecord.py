"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-10-7 下午10:15
# @FileName: image_to_tfrecord.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
import io
from PIL import Image
from get_image2label_id import get_image2label_id

flags = tf.app.flags
flags.DEFINE_string('images_dir',
                    '../../data/17flowers',
                    'path of image')
flags.DEFINE_string('record_path',
                    '../../data/17flowers/train.record',
                    'path of TFRecord')
FLAGS = flags.FLAGS


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def process_image_channels(image):
    processed_flag = False
    # process the 4 channels .png
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r, g, b))
        processed_flag = True
    # process the channel image
    elif image.mode != 'RGB':
        image = image.convert("RGB")
        processed_flag = True
    return image, processed_flag


def process_image_reshape(image, resize):
    width, height = image.size
    if resize is not None:
        if width > height:
             width = int(width * resize / height)
             height = resize
        else:
            width = resize
            height = int(height * resize / width)
        image = image.resize((width, height), Image.ANTIALIAS)
    return image


def create_tf_example(image_path, label, resize=None):
    with tf.gfile.GFile(image_path, 'rb') as fi:
        encode_jpg = fi.read()
    encode_jpg_io = io.BytesIO(encode_jpg)
    image = Image.open(encode_jpg_io)
    # process png pic with four channels
    image, processed_flag = process_image_channels(image)
    # reshape image
    image = process_image_reshape(image, resize)
    if processed_flag is True or resize is not None:
        bytes_io = io.BytesIO()
        image.save(bytes_io, format='JPEG')
        encoded_jpg = bytes_io.getvalue()
    width, height = image.size
    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': bytes_feature(encode_jpg),
                'image/format': bytes_feature(b'jpg'),
                'image/class/label': int64_feature(label),
                'image/height': int64_feature(height),
                'image/width': int64_feature(width)
            }
        ))
    return tf_example


def write_tfrecord(annotation_dict, record_path, resize=None):
    num_tf_example = 0
    writer = tf.python_io.TFRecordWriter(record_path)
    for image_path, label in annotation_dict.items():
        if not tf.gfile.GFile(image_path):
            print("{} does not exist".format(image_path))
        tf_example = create_tf_example(image_path, label, resize)
        writer.write(tf_example.SerializeToString())
        num_tf_example += 1
        if num_tf_example % 100 == 0:
            print("Create %d TF_Example" % num_tf_example)
    writer.close()
    print("{} tf_examples has been created successfully, which are saved in {}".format(num_tf_example, record_path))


def get_image_class2number():
    image_class2number = {}
    for i in range(1, 18):
        image_class2number.setdefault(str(i), i)
    return image_class2number


def display_tfrecord(tfrecord_file):
    item = next(tf.python_io.tf_record_iterator(tfrecord_file))
    print(tf.train.Example.FromString(item))


def main(_):
    image_class_name2number = get_image_class2number()
    images_dir = FLAGS.images_dir
    record_path = FLAGS.record_path
    image2label_id = get_image2label_id(images_dir, image_class_name2number)
    write_tfrecord(image2label_id, record_path)
    # display_tfrecord(record_path)


if __name__ == '__main__':
    tf.app.run()
