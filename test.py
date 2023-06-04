import os
import time
import string
import argparse
import re
import validators

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance

from train import validation
from utils import TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollate, LmdbDataset
from model import Model
from utils import get_args
from collections import OrderedDict


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(opt):
    """ model configuration """

    converter = TokenLabelConverter(opt)

    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)

    print('model input parameters', opt.imgH, opt.imgW, opt.num_class, opt.batch_max_length)
    model = torch.nn.DataParallel(model).to(device)


    # load model
    print('loading pretrained model from %s' % opt.saved_model)

    if validators.url(opt.saved_model):
        model.load_state_dict(torch.hub.load_state_dict_from_url(opt.saved_model, progress=True, map_location=device))
    else:
        # # 读取参数
        # pretrained_dict = torch.load(opt.saved_model, map_location=device)
        # model_dict = model.state_dict()
        #
        # # 将pretrained_dict里不属于model_dict的键剔除掉
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #
        # # 更新现有的model_dict
        # model_dict.update(pretrained_dict)
        #
        # # 加载我们真正需要的state_dict
        # model.load_state_dict(model_dict)
        checkpoint = torch.load(opt.saved_model, map_location=device)
        # for name in checkpoint.keys():
        #     print(name)
        model.load_state_dict(checkpoint)
        # checkpoint = torch.load(opt.saved_model)
        # model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})

    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)





    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

    """ setup loss """

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    """ evaluation """
    model.eval()
    opt.eval = True
    with torch.no_grad():
        # if opt.benchmark_all_eval:  # evaluation with 10 benchmark evaluation datasets
        #     benchmark_all_eval(model, criterion, converter, opt)
        # else:
            log = open(f'./result/{opt.exp_name}/log_evaluation.txt', 'a')
            AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
            eval_data, eval_data_log = hierarchical_dataset(root=opt.eval_data, opt=opt, mode='eval')
            evaluation_loader = torch.utils.data.DataLoader(
                eval_data, batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.workers,
                collate_fn=AlignCollate_evaluation, pin_memory=True)
            _, accuracy_by_best_model, _, _, _, _, _, _ = validation(
                model, criterion, evaluation_loader, converter, opt)
            log.write(eval_data_log)
            print(f'Acc:{accuracy_by_best_model:0.3f}')
            log.write(f'Acc:{accuracy_by_best_model:0.3f}\n')
            log.close()


if __name__ == '__main__':
    opt = get_args(is_train=False)

    """ vocab / character number configuration """

    # opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
    opt.character = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{}~ '
    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    test(opt)