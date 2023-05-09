"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-10-19 上午12:00
# @FileName: nlp_sm_est_trainer.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
import os
from src.models.nlp.sm.dssm import DssmModel
from src.models.nlp.sm.clsm import ClsmModel


flags = tf.flags
flags.DEFINE_string("task_type", "classification", "task type: cls or reg")
flags.DEFINE_string("model_name", "dssm", "nlp model")
flags.DEFINE_boolean("is_cluster", False, "Whether run on cluster during training model or not")
# flags.DEFINE_boolean("export_query_model", False, "Whether export model or not")
# flags.DEFINE_boolean("export_doc_model", False, "Whether export model or not")
flags.DEFINE_boolean("is_train", False, "Whether training model or not")

flags.DEFINE_string("input", "data/nlp/sm/train.tfrecord", "input data")
flags.DEFINE_string("test_file_path", "data/nlp/sm/test.tfrecord", "test data")
flags.DEFINE_string("job_name", "", "job name of ps")
flags.DEFINE_string("ps_master", "", "comma-separated list of hostname:port pairs")
flags.DEFINE_string("ps_worker", "", "comma-separated list of hostname:port pairs")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")

flags.DEFINE_string("output", "output", "output path of model.")

flags.DEFINE_integer("rnn_hidden_size", 64, "rnn_hidden_size")
flags.DEFINE_integer("query_max_length", 60, "max token length of query text")
flags.DEFINE_integer("doc_max_length", 200, "max token length of doc text")
flags.DEFINE_integer("token_embedding_size", 128, "token embedding size")
flags.DEFINE_string('layers', "128,64", "layers of nn")
flags.DEFINE_integer("last_hidden_size", 64, "last hidden size")

flags.DEFINE_float("lr", 0.001, "learning rate")
flags.DEFINE_float("dropout", 1.0, "keep rate")
flags.DEFINE_integer("token_vocab_size", 10000, "vocab size of token")
flags.DEFINE_integer("neg_num", 5, "negative samples")
flags.DEFINE_integer("epochs", 2, "training epochs")
flags.DEFINE_integer("train_steps", 1000, "training steps")
flags.DEFINE_integer("batch_size", 128, "training batch size")
flags.DEFINE_integer("display_steps", 100, "saving checkpoint of steps")
args = flags.FLAGS


class NlpSmEstTrainer(object):

    def __init__(self, task_type):
        self.task_type = task_type
        self.classifier = None

    def train(self, model):
        """
        only train model, not execute stage of eval model
        :param model:
        :return:
        """
        self.classifier = tf.estimator.Estimator(
            model_fn=model.model_fn,
            config=tf.estimator.RunConfig(
                model_dir=model.output,
                save_checkpoints_steps=model.display_steps,
                keep_checkpoint_max=3
            ),
            params={}
        )

        self.classifier.train(
            input_fn=lambda: model.input_fn(
                model.input,
                tf.estimator.ModeKeys.TRAIN,
                model.batch_size
            ),
            max_steps=1000
        )
        feed_dict = {"query": model.query_max_length}
        model.export_tf_model(self.classifier, feed_dict, os.path.join(model.output, 'query'))
        # feed_dict = {"doc": model.doc_max_length}
        # model.export_tf_model(self.classifier, feed_dict, os.path.join(model.output, 'doc'))

    def train_and_eval(self, model):
        """ train and eval model during training """
        self.classifier = tf.estimator.Estimator(
            model_fn=model.model_fn,
            config=tf.estimator.RunConfig(
                model_dir=model.output,
                save_checkpoints_steps=model.display_steps,
                keep_checkpoint_max=3
            ),
            params={}
        )

        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: model.input_fn(
                model.input,
                tf.estimator.ModeKeys.TRAIN,
                model.batch_size
            ),
            max_steps=1000
        )

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: model.input_fn(
                model.test_file_path,
                tf.estimator.ModeKeys.EVAL,
                model.batch_size
            ),
            start_delay_secs=60,
            throttle_secs=30,
            steps=None
        )

        tf.estimator.train_and_evaluate(self.classifier, train_spec, eval_spec)
        # feed_dict = {"query": model.query_max_length}
        # model.export_tf_model(self.classifier, feed_dict, os.path.join(model.output, 'query'))
        # feed_dict = {"doc": model.doc_max_length}
        # model.export_tf_model(self.classifier, feed_dict, os.path.join(model.output, 'doc'))


def main(_):
    """模型训练程序运行入口"""
    # args = tf_arg_parse()
    task_type = args.task_type
    model_name = args.model_name

    if model_name == 'dssm':
        model = DssmModel(args, task_type)
    elif model_name == 'clsm':
        model = ClsmModel(args, task_type)
    else:
        print('not support!')
        return

    if tf.gfile.Exists(model.output):
        tf.gfile.DeleteRecursively(model.output)
    print("==========total train steps=========: ", model.train_steps * model.epochs)
    trainer = NlpSmEstTrainer(task_type)
    trainer.train(model)


if __name__ == '__main__':
    tf.compat.v1.app.run()
