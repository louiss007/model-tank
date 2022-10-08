"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-10-7 下午10:17
# @FileName: read_tfrecord_for_cv.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('image_tfrecord_path',
                    '../../data/tfrecord/cv/out/train.record',
                    'path of tfrecord file')
flags.DEFINE_integer('resize_height', 800, 'resize height of image')
flags.DEFINE_integer('resize_width', 800, 'resize width of image')
FLAG = flags.FLAGS
slim = tf.contrib.slim


def display_data(image, resized_image, label, height, width):
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(10):
            print("___________________image({})__________________".format(i))
            p_image, p_resized_image, p_label, p_height, p_width = sess.run(
                [image, resized_image, label, height, width])
            print("resized_image shape is: ", p_resized_image.shape)
            print("image shape is: ", p_image.shape)
            print("image label is: ", p_label)
            print("image height is: ", p_height)
            print("image width is: ", p_width)
        coord.request_stop()
        coord.join(threads)


def reshape_same_size(image, output_height, output_width):
    """Resize images by fixed sides.

    Args:
        image: A 3-D image `Tensor`.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.

    Returns:
        resized_image: A 3-D tensor containing the resized image.
    """
    output_height = tf.convert_to_tensor(output_height, dtype=tf.int32)
    output_width = tf.convert_to_tensor(output_width, dtype=tf.int32)

    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_nearest_neighbor(
        image,
        [output_height, output_width],
        align_corners=False
    )
    resized_image = tf.squeeze(resized_image)
    return resized_image


def read_tfrecord(tfrecord_path, num_samples=14635, num_classes=7, resize_height=800, resize_width=800):
    image_features = {
        'image/encoded': tf.FixedLenFeature([], default_value='', dtype=tf.string, ),
        'image/format': tf.FixedLenFeature([], default_value='jpeg', dtype=tf.string),
        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=0),
        'image/height': tf.FixedLenFeature([], tf.int64, default_value=0),
        'image/width': tf.FixedLenFeature([], tf.int64, default_value=0)
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(image_key='image/encoded', format_key='image/format', channels=3),
        'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
        'height': slim.tfexample_decoder.Tensor('image/height', shape=[]),
        'width': slim.tfexample_decoder.Tensor('image/width', shape=[])
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(image_features, items_to_handlers)

    labels_to_names = None
    items_to_descriptions = {
        'image': 'An image with shape image_shape.',
        'label': 'A single integer between 0 and 9.'
    }
    dataset = slim.dataset.Dataset(
        data_sources=tfrecord_path,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions=None,
        num_classes=num_classes,
    )

    provider = slim.dataset_data_provider.DatasetDataProvider(dataset=dataset,
                                                              num_readers=3,
                                                              shuffle=True,
                                                              common_queue_capacity=256,
                                                              common_queue_min=128,
                                                              seed=None)
    image, label, height, width = provider.get(['image', 'label', 'height', 'width'])
    resized_image = tf.squeeze(tf.image.resize_bilinear([image], size=[resize_height, resize_width]))
    return resized_image, label, image, height, width


def main():
    resized_image, label, image, height, width = read_tfrecord(tfrecord_path=FLAG.tfrecord_path,
                                                               resize_height=FLAG.resize_height,
                                                               resize_width=FLAG.resize_width)
    # resized_image = reshape_same_size(image, FLAG.resize_height, FLAG.resize_width)
    # resized_image = tf.squeeze(tf.image.resize_bilinear([image], size=[FLAG.resize_height, FLAG.resize_width]))
    display_data(image, resized_image, label, height, width)


if __name__ == '__main__':
    main()
