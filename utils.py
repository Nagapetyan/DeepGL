import numpy as np
import torch

from model import DeepGL

from logger import Logger

import os


def save_parameters(args, run_name):
    with open(os.path.join(args.log_path, run_name)+'/parameters.txt', 'w') as f:
        f.write('num_blocks {}, lr {}, beta1 {} beta2 {}, batch_size {} gamma  {} scheduler_step {}'.format(
            args.num_blocks, args.lr, args.beta1, args.beta2, args.batch_size, 
            args.gamma, args.scheduler_step
        ))


def prepare_directories(args, run_name):
    if not os.path.isdir(args.data_path):
        raise Exception("Invalid data path. No such directory")

    if not os.path.isdir(args.log_path):
        os.makedirs(args.log_path)

    if args.pretrained_path:
        if not os.path.isdir(args.pretrained_path) or \
                not os.path.isdir(os.path.join(args.pretrained_path, 'states')):
            raise Exception("Invalid path. No such directory with pretrained model")

    else:
        exp_path = os.path.join(args.log_path, run_name)
        os.makedirs(exp_path)
        os.makedirs(os.path.join(exp_path, 'samples'))
        os.makedirs(os.path.join(exp_path, 'states'))
        os.makedirs(os.path.join(exp_path, 'tensorboard_logs'))


def build_model(args):
    model = DeepGL(args.num_blocks)
    if args.pretrained_path:
        model.load_state_dict(torch.load(
            os.path.join(args.pretrained_path, 'samples') + '/' + str(args.load_step) + '.pt'))

    return model


def prepare_logger(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    logger = Logger(path)
    return logger

