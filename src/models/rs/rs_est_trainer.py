"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-11-19 上午10:44
# @FileName: rs_est_trainer.py
# @Email   : quant_master2000@163.com
======================
"""
import tensorflow as tf
from utils.arg_parse import arg_parse
from src.models.rs.mlp import MLP
from src.models.rs.wnd import WideAndDeep
from src.models.rs.deepfm import DeepFm


class RsEstTrainer(object):

    def __init__(self, task_type):
        self.task_type = task_type
        self.classifier = None

    def train(self, model, params=None):
        """
        only train model, not execute stage of eval model
        :param model:
        :return:
        """
        self.classifier = tf.estimator.Estimator(
            model_fn=model.model_fn,
            config=tf.estimator.RunConfig(
                model_dir=model.output,
                save_checkpoints_steps=model.display_step,
                keep_checkpoint_max=3
            ),
            params=params
        )

        self.classifier.train(
            input_fn=lambda: model.input_fn(
                model.input,
                model.batch_size
            ),
            max_steps=1000
        )
        # features = {}
        # model.export_tf_model(self.classifier, features=features, model_dir=model.output)
        # feed_dict = {"doc": model.doc_max_length}
        # model.export_tf_model(self.classifier, feed_dict, os.path.join(model.output, 'doc'))

    def train_and_eval(self, model, params=None):
        """ train and eval model during training """
        self.classifier = tf.estimator.Estimator(
            model_fn=model.model_fn,
            config=tf.estimator.RunConfig(
                model_dir=model.output,
                save_checkpoints_steps=model.display_step,
                keep_checkpoint_max=3
            ),
            params=params
        )

        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: model.input_fn(
                model.input,
                model.batch_size
            ),
            max_steps=100000
        )

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: model.input_fn(
                model.eval_file,
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
    task_description = 'Rs Model Train Task With tf.Estimator Mode!'
    parser = arg_parse(task_description)
    args = parser.parse_args()
    print(args)
    task_type = args.task_type
    model_name = args.model_name

    params = None
    if model_name == 'mlp':
        model = MLP(args, task_type)
    elif model_name == 'wnd':
        model = WideAndDeep(args, task_type)
    elif model_name == 'deepfm':
        params = {
            "field_size": 39,
            # "feature_size": 117581,
            "feature_size": 100000,
            "embedding_size": args.embedding_size,
            "l2_reg": 0.0001
        }
        model = DeepFm(args, task_type)

    else:
        print('not support!')
        return

    if tf.gfile.Exists(model.output):
        tf.gfile.DeleteRecursively(model.output)
    # print("==========total train steps=========: ", model.train_steps * model.epochs)
    trainer = RsEstTrainer(task_type)
    trainer.train_and_eval(model, params)


if __name__ == '__main__':
    tf.compat.v1.app.run()
