seed = 123
import numpy as np
from sympy import arg
np.random.seed(seed)
import random as rn
rn.seed(seed)
import os
os.environ['PYTHONHASHSEED'] = str(seed)
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from utils.config import args
import time
from datetime import datetime

import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

cudnn.benchmark = True

import nets as models
# from utils.preprocess import *
from utils.bar_show import progress_bar
import pdb
from src.cmdataset import CMDataset
import scipy
import scipy.spatial
import torch.nn as nn
import src.utils as utils
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCESoftmaxLoss
from NCE.NCECriterion import ModifiedNCESoftmaxLoss
from torch.nn.utils.clip_grad import clip_grad_norm
from NCE import ce_loss

from NCE.sdm_loss import *

# --pretrain --arch resnet18

device_ids = [0, 1]
teacher_device_id = [0, 1]
best_acc = 0  # best test accuracy
start_epoch = 0

args.log_dir = os.path.join(args.root_dir, 'logs', args.log_name)
args.ckpt_dir = os.path.join(args.root_dir, 'ckpt', args.pretrain_dir)

os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.ckpt_dir, exist_ok=True)

def main():
    print('===> Preparing data ..')
        # build data
    train_dataset = CMDataset(
        args.data_name,
        return_index=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    retrieval_dataset = CMDataset(
        args.data_name,
        partition='retrieval'
    )
    retrieval_loader = torch.utils.data.DataLoader(
        retrieval_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_dataset = CMDataset(
        args.data_name,
        partition='test'
    )
    query_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    print('===> Building ResNet..')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if 'fea' in args.data_name:
        image_model = models.__dict__['ImageNet'](y_dim=train_dataset.imgs.shape[1], bit=args.bit, num_classes=args.classes, hiden_layer=args.num_hiden_layers[0]).cuda()
        backbone = None
    else:
        backbone = models.__dict__[args.arch](pretrained=args.pretrain, feature=True).cuda()
        fea_net = models.__dict__['ImageNet'](y_dim=4096 if 'vgg' in args.arch.lower() else (512 if args.arch == 'resnet18' or args.arch == 'resnet34' else 2048), bit=args.bit, hiden_layer=args.num_hiden_layers[0]).cuda()
        image_model = nn.Sequential(backbone, fea_net)
    text_model = models.__dict__['TextNet'](y_dim=train_dataset.text_dim, bit=args.bit, num_classes=args.classes, hiden_layer=args.num_hiden_layers[1]).cuda()

    parameters = list(image_model.parameters()) + list(text_model.parameters())
    wd = args.wd
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=wd)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=wd)
    if args.ls == 'cos':
        lr_schedu = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, eta_min=0, last_epoch=-1)
    else:
        lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90, 120], gamma=0.1)

    summary_writer = SummaryWriter(args.log_dir)

    if args.resume:
        ckpt = torch.load(os.path.join(args.ckpt_dir, args.resume))
        image_model.load_state_dict(ckpt['image_model_state_dict'])
        text_model.load_state_dict(ckpt['text_model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        print('===> Load last checkpoint data')
    else:
        start_epoch = 0
        print('===> Start from scratch')

    def set_train(is_warmup=False):
        image_model.train()
        if is_warmup and backbone:
            backbone.eval()
            backbone.requires_grad_(False)
        elif backbone:
            backbone.requires_grad_(True)
        text_model.train()

    def set_eval():
        image_model.eval()
        text_model.eval()

    criterion = utils.ContrastiveLoss(args.margin, shift=args.shift)
    n_data = len(train_loader.dataset)  # 包含了训练集的所有样本
    contrast = NCEAverage(args.bit, n_data, args.K, args.T, args.momentum, args.classes)
    criterion_contrast = NCESoftmaxLoss()
    criterion_contrast1 = ModifiedNCESoftmaxLoss()
    contrast = contrast.cuda()
    criterion_contrast = criterion_contrast.cuda()
    # 添加分类损失
    criterion_ce = ce_loss.CELoss()
    # 添加SDM损失

    def train(epoch):
        print('\nEpoch: %d / %d' % (epoch, args.max_epochs))
        set_train(epoch < args.warmup_epoch)
        train_loss, correct, total = 0., 0., 0.
        for batch_idx, (idx, images, texts, labels) in enumerate(train_loader):
            images, texts, idx, labels_b = [img.cuda() for img in images], [txt.cuda() for txt in texts], [
                idx.cuda()], torch.stack([lab.cuda() for lab in labels])

            # 转换 labels 为类别索引
            labels = torch.argmax(labels_b, dim=1)

            images_outputs, image_logits = [], []
            for im in images:
                output, logit = image_model(im)
                images_outputs.append(output)
                image_logits.append(logit)

            texts_outputs, text_logits = [], []
            for txt in texts:
                output, logit = text_model(txt.float())
                texts_outputs.append(output)
                text_logits.append(logit)

            out_l, out_ab, flag = contrast(labels_b, torch.cat(images_outputs), torch.cat(texts_outputs), torch.cat(idx * len(images)),
                                     args.classes, args.pos_num, args.neg_num, epoch=epoch - args.warmup_epoch)

            if flag <= 0:
                l_loss = criterion_contrast(out_l)
                ab_loss = criterion_contrast(out_ab)
            else:
                # 取出索引，新添加
                pos_out_l = out_l[:, :args.pos_num]
                neg_out_l = out_l[:, args.pos_num:]
                pos_out_ab = out_ab[:, :args.pos_num]
                neg_out_ab = out_ab[:, args.pos_num:]
                l_loss = criterion_contrast1(pos_out_l, neg_out_l)
                ab_loss = criterion_contrast1(pos_out_ab, neg_out_ab)



            ce_loss_image = criterion_ce(torch.cat(image_logits), labels_b)
            ce_loss_text = criterion_ce(torch.cat(text_logits), labels_b)
            # # 定义分类损失
            # Lce = ce_loss_text + ce_loss_image
            # 定义sdm损失
            Lsdm = compute_sdm(images_outputs, texts_outputs, labels_b, args.logit_scale)
            # Lsdm_intra = compute_in_modal_sdm(images_outputs, labels_b, logit_scale)
            # Lsdm_intra_txt = compute_in_modal_sdm(texts_outputs, labels_b, logit_scale)
            Lc = l_loss + ab_loss
            Lr = criterion(torch.cat(images_outputs), torch.cat(texts_outputs))
            loss = Lc * args.alpha + Lr * (1. - args.alpha-args.yita) + args.yita * Lsdm

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(parameters, 1.)
            optimizer.step()
            train_loss += loss.item()
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | LR: %g'
                         % (train_loss / (batch_idx + 1), optimizer.param_groups[0]['lr']))

            if batch_idx % args.log_interval == 0:
                print('Loss/train/lr', train_loss / (batch_idx + 1), epoch * len(train_loader) + batch_idx,
                      optimizer.param_groups[0]['lr'])
                summary_writer.add_scalar('Loss/train', train_loss / (batch_idx + 1),
                                          epoch * len(train_loader) + batch_idx)
                summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'],
                                          epoch * len(train_loader) + batch_idx)

    def eval(data_loader):
        set_eval()
        imgs, txts, labs = [], [], []
        imgs_embed_npy = []
        txts_embed_npy = []
        with torch.no_grad():
            for batch_idx, (images, texts, targets) in enumerate(data_loader):
                images, texts, targets = [img.cuda() for img in images], [txt.cuda() for txt in texts], targets.cuda()

                # images_outputs, _ = [image_model(im) for im in images]
                # texts_outputs, _ = [text_model(txt.float()) for txt in texts]
                # 修改为
                images_outputs = []

                for im in images:
                    output, _ = image_model(im)
                    images_outputs.append(output)

                texts_outputs = []
                txts_1024_outputs = []
                for txt in texts:
                    output, _ = text_model(txt.float())
                    texts_outputs.append(output)

                imgs += images_outputs
                txts += texts_outputs
                labs.append(targets)

            imgs = torch.cat(imgs).sign_().cpu().numpy()
            txts = torch.cat(txts).sign_().cpu().numpy()
            labs = torch.cat(labs).cpu().numpy()


        return imgs, txts, labs

    def test(epoch, is_eval=True):
        # pass
        global best_acc
        set_eval()
        data_embed_collect = []
        # switch to evaluate mode
        (retrieval_imgs, retrieval_txts, retrieval_labs) = eval(retrieval_loader)
        if is_eval:
            query_imgs, query_txts, query_labs = retrieval_imgs[0: 2000], retrieval_txts[0: 2000], retrieval_labs[0: 2000]
            retrieval_imgs, retrieval_txts, retrieval_labs = retrieval_imgs[0: 2000], retrieval_txts[0: 2000], retrieval_labs[0: 2000]
        else:
            (query_imgs, query_txts, query_labs) = eval(query_loader)


        i2t = fx_calc_map_multilabel_k(retrieval_txts, retrieval_labs, query_imgs, query_labs, k=0, metric='hamming')
        t2i = fx_calc_map_multilabel_k(retrieval_imgs, retrieval_labs, query_txts, query_labs, k=0, metric='hamming')


        avg = (i2t + t2i) / 2.
        print('%s\nImg2Txt: %g \t Txt2Img: %g \t Avg: %g' % ('Evaluation' if is_eval else 'Test',i2t, t2i, (i2t + t2i) / 2.))

        if avg >= best_acc:
            print('Saving..')
            state = {
                'image_model_state_dict': image_model.state_dict(),
                'text_model_state_dict': text_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'Avg': avg,
                'Img2Txt': i2t,
                'Txt2Img': t2i,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(args.ckpt_dir, '%s_%d_best_checkpoint.t7' % (args.arch, args.bit)))
            best_acc = avg
            print('best_acc:',best_acc)
        return i2t, t2i

    lr_schedu.step(start_epoch)
    for epoch in range(start_epoch, args.max_epochs):
        train(epoch)
        lr_schedu.step(epoch)
        if epoch % 10 == 0:
            i2t, t2i = test(epoch)
            avg = (i2t + t2i) / 2.
            if avg == best_acc:
                image_model_state_dict = image_model.state_dict()
                image_model_state_dict = {key: image_model_state_dict[key].clone() for key in image_model_state_dict}
                text_model_state_dict = text_model.state_dict()
                text_model_state_dict = {key: text_model_state_dict[key].clone() for key in text_model_state_dict}

    chp = torch.load(os.path.join(args.ckpt_dir, '%s_%d_best_checkpoint.t7' % (args.arch, args.bit)))
    image_model.load_state_dict(image_model_state_dict)
    text_model.load_state_dict(text_model_state_dict)
    test(chp['epoch'], is_eval=False)
    #test(epoch,is_eval=False)
    summary_writer.close()
    # pdb.set_trace()

def fx_calc_map_multilabel_k(retrieval, retrieval_labels, query, query_label, k=0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(query, retrieval, metric)
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i].reshape(-1)
        # order = ord[i][:k].reshape(-1)
        tmp_label = (np.dot(retrieval_labels[order], query_label[i]) > 0)
        if tmp_label.sum() > 0:
            prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
            total_pos = float(tmp_label.sum())
            if total_pos > 0:
                res += [np.dot(tmp_label, prec) / total_pos]
    return np.mean(res)













if __name__ == '__main__':
    main()

