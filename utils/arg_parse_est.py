"""
======================
# -*-coding: utf8-*-
# @Author  : louiss007
# @Time    : 22-10-22 下午5:40
# @FileName: arg_parse_est.py
# @Email   : quant_master2000@163.com
======================
"""

import argparse


def arg_parse(task_description):
    # Training settings
    parser = argparse.ArgumentParser(description='{}'.format(task_description))
    parser.add_argument('--task-type', type=str, default='classification', metavar='c/r',
                        help='input task type for training (default: classification)')
    parser.add_argument('--model-name', type=str, default='fnn', metavar='model',
                        help='input model type for training (default: fnn)')

    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--dropout', type=float, default=1.0, metavar='F',
                        help='input batch size for testing (default: 1.0)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--input-size', type=int, default=28, metavar='IN',
                        help='size of input layer (default: 28)')
    parser.add_argument('--layers', type=str, default='784,128,256,32', metavar='layer',
                        help='hidden layer of model (default: 128,256,32)')
    parser.add_argument('--nclass', type=int, default=10, metavar='NC',
                        help='size of output layer (default: 10)')

    parser.add_argument('--input', type=str, required=True, metavar='input',
                        help='input training data file')
    parser.add_argument('--eval-file', type=str, required=True, metavar='eval',
                        help='input eval data file')
    parser.add_argument('--output', type=str, required=True, metavar='output',
                        help='output path for saving model')
    parser.add_argument('--display-step', type=int, default=200, metavar='N',
                        help='how many steps to display training process (default: 200)')
    return parser
