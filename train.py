import os
import sys
import time
import random
import string
import argparse
import re
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
from utils import Averager, TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from utils import get_args
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR, ReduceLROnPlateau
from nltk.metrics.distance import edit_distance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validation(model, criterion, evaluation_loader, converter, opt):
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        target = converter.encode(labels)
        start_time = time.time()
        preds = model(image, text=target, seqlen=converter.batch_max_length)
        _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
        preds_index = preds_index.view(-1, converter.batch_max_length)
        forward_time = time.time() - start_time
        cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))
        # ctc_loss = torch.nn.CTCLoss(reduction='mean').to(device)
        # input_lengths = (torch.ones(64, dtype=torch.long) * 27)
        # log_probs = preds.transpose(0, 1).log_softmax(2).detach().requires_grad_()
        # target_length = []
        # for label in labels:
        #     target_length.append(len(label))
        # target_length = torch.Tensor(target_length).int()
        # cost2 = ctc_loss(log_probs, target, input_lengths, target_length)
        # cost = (cost2 + cost1) / 2
        length_for_pred = torch.IntTensor([converter.batch_max_length - 1] * batch_size).to(device)
        preds_str = converter.decode(preds_index[:, 1:], length_for_pred)
        infer_time += forward_time
        valid_loss_avg.add(cost)
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if opt.Transformer:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]
                pred_max_prob = pred_max_prob[:pred_EOS]
            if opt.sensitive and opt.data_filtering_off:
                pred = pred.lower()
                gt = gt.lower()
                alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)
            if pred == gt:
                n_correct += 1
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0
            confidence_score_list.append(confidence_score)
    current_accuracy = n_correct / float(length_of_data) * 100
    current_norm_ED = norm_ED / float(length_of_data)
    return valid_loss_avg.val(), current_accuracy, current_norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data

def train(opt):
    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    converter = TokenLabelConverter(opt)
    opt.num_class = len(converter.character)
    opt.eval = False
    train_dataset = Batch_Balanced_Dataset(opt)
    log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
    opt.eval = True
    opt.data_filtering_off = True
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
    eval_data_path = os.path.join(opt.valid_data, opt.select_data[0])
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt, mode='valid')
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers,
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()

    converter = TokenLabelConverter(opt)

    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))
    # model.load_state_dict(torch.load('dcstr_tiny_patch16_224_aug.pth', map_location=device), False)

    # set loss

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    # print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    scheduler = None
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    scheduler = CosineAnnealingLR(optimizer, T_max=opt.num_iter)
    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter
    while(True):
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)

        target = converter.encode(labels)
        preds = model(image, text=target, seqlen=converter.batch_max_length)
        cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        # input_lengths = (torch.ones(64) * 27).int()
        # log_probs = preds.transpose(0, 1).log_softmax(2).detach().requires_grad_()
        # target_length = []
        # for label in labels:
        #     target_length.append(len(label))
        # target_length = torch.Tensor(target_length).int()
        # cost2 = ctc_loss(log_probs, target, input_lengths, target_length)
        # cost = (cost2 + cost1) / 2
        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        loss_avg.add(cost)

        if (iteration + 1) % opt.valInterval == 0 or iteration == 0:
            elapsed_time = time.time() - start_time
            with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a', encoding='utf-8') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, criterion, valid_loader, converter, opt)
                model.train()

                loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()
                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_norm_ED.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'
                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                    if opt.Transformer:
                        pred = pred[:pred.find('[s]')]
                    if opt.sensitive and opt.data_filtering_off:
                        pred = pred.lower()
                        gt = gt.lower()
                        alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                        out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                        pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                        gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)
                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')
        if (iteration + 1) % 1e+4 == 0:
            torch.save(
                model.state_dict(), f'./saved_models/{opt.exp_name}/iter_{iteration+1}.pth')
        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1
        if scheduler is not None:
            scheduler.step()


if __name__ == '__main__':
    opt = get_args()
    opt.exp_name = f'{opt.TransformerModel}' + f'-Seed{opt.manualSeed}'
    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)
    # opt.character = string.printable
    char_set = open('dict/arabic_dict.txt', 'r', encoding='utf-8').readlines()
    char_set = ''.join([ch.strip('\n') for ch in char_set][0:])
    opt.character = char_set
    # opt.character = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|~ '
    # 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{}~
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    train(opt)