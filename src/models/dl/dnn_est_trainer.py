"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-10-22 下午9:57
# @FileName: dnn_est_trainer.py
# @Email   : quant_master2000@163.com
# Reference:
# [1]. https://www.cnblogs.com/hansjorn/p/10646346.html
======================
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from utils.arg_parse import arg_parse
from src.models.dl.tf_estimator.fnn import FnnModel
from src.models.dl.tf_estimator.cnn import CnnModel

print('*******tf_version:%s********' % tf.__version__)
# tf.logging.set_verbosity(tf.logging.INFO)


# Our application logic will be added here
class DnnEstTrainer(object):

    def __init__(self, task_type, train_file, eval_file):
        self.task_type = task_type
        self.train_data = train_file
        self.eval_data = eval_file

    def train(self, model):
        # Create the Estimator
        classifier = tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=model.output
        )

        # Set up logging for predictions
        # tensors_to_log = {"probabilities": "softmax_tensor"}
        # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

        # Train the model
        # train_input_fn = tf.estimator.inputs.numpy_input_fn(
        #     x={"x": self.train_data},
        #     y=self.train_label,
        #     batch_size=model.batch_size,
        #     num_epochs=model.epochs,
        #     shuffle=True
        # )

        classifier.train(
            input_fn=lambda: model.load_data(self.train_data),
            steps=model.train_steps,
            # hooks=[logging_hook]
            hooks=None
        )

    def train_and_eval(self, model):
        # Create the Estimator
        classifier = tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=model.output
        )

        # Set up logging for predictions
        # tensors_to_log = {"probabilities": "softmax_tensor"}
        # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

        # Train the model
        # train_input_fn = tf.estimator.inputs.numpy_input_fn(
        #     x={"x": self.train_data},
        #     y=self.train_label,
        #     batch_size=model.batch_size,
        #     num_epochs=model.epochs,
        #     shuffle=True
        # )

        classifier.train(
            input_fn=lambda: model.load_data(self.train_data),
            steps=model.train_steps,
            # hooks=[logging_hook]
            hooks=None
        )

        # Evaluate the model and print results
        # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        #     x={"x": self.eval_data},
        #     y=self.eval_label,
        #     num_epochs=1,
        #     shuffle=False
        # )
        eval_results = classifier.evaluate(input_fn=lambda: model.load_data(self.eval_data))
        print(eval_results)


def main(_):
    # direct run from pycharm, couldn't call GPU,
    # must run in terminal command line format!!!

    """模型训练程序运行入口"""
    task_description = 'Dnn Model Train Task With Session Mode!'
    parser = arg_parse(task_description)
    args = parser.parse_args()
    print(args)
    task_type = args.task_type
    model_name = args.model_name

    if model_name == 'fnn':
        model = FnnModel(args, task_type)
    elif model_name == 'cnn':
        model = CnnModel(args, task_type)
    else:
        print('model %s not supported!' % model_name)
        return

    print('Model %s is called' % model.__class__.__name__)

    # Load training and eval data
    train_file = os.path.join(model.input, 'train.tfrecord')
    eval_file = os.path.join(model.input, 'test.tfrecord')
    # train_data, train_label = model.load_data(train_file)
    # eval_data, eval_label = model.load_data(eval_file)
    # print(train_data)
    import numpy as np
    # Load training and eval data
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images  # Returns np.array
    # train_label = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images  # Returns np.array
    # eval_label = np.asarray(mnist.test.labels, dtype=np.int32)
    trainer = DnnEstTrainer(task_type, train_file, eval_file)
    trainer.train_and_eval(model)


if __name__ == "__main__":
    tf.app.run()
