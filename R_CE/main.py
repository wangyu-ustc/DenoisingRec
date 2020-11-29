import os
import time
import copy
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from FocalLoss import FocalLoss
# from tensorboardX import SummaryWriter

import model
import evaluate
import data_utils
from resnet import VNet
from loss import loss_function

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    type=str,
                    help='dataset used for training, options: amazon_book, yelp, adressa',
                    default='amazon_book')
parser.add_argument("--tb_dir",
                    type=str,
                    help="where to put tensorboard record",
                    default='./models')
parser.add_argument('--model',
                    type=str,
                    help='model used for training. options: GMF, NeuMF-end',
                    default='GMF')
parser.add_argument('--alpha',
                    type=float,
                    default=0.2,
                    help='hyperparameter in loss function')
parser.add_argument("--lr",
                    type=float,
                    default=0.001,
                    help="learning rate")
parser.add_argument("--dropout",
                    type=float,
                    default=0.0,
                    help="dropout rate")
parser.add_argument("--batch_size",
                    type=int,
                    default=1024,
                    help="batch size for training")
parser.add_argument("--epochs",
                    type=int,
                    default=10,
                    help="training epoches")
parser.add_argument("--eval_freq",
                    type=int,
                    default=200,
                    help="the freq of eval")
parser.add_argument("--top_k",
                    type=list,
                    default=[50, 100],
                    help="compute metrics@top_k")
parser.add_argument("--factor_num",
                    type=int,
                    default=32,
                    help="predictive factors numbers in the model")
parser.add_argument("--num_layers",
                    type=int,
                    default=3,
                    help="number of layers in MLP model")
parser.add_argument("--num_ng",
                    type=int,
                    default=1,
                    help="sample negative items for training")
parser.add_argument("--out",
                    default=True,
                    help="save model or not")
parser.add_argument("--gpu",
                    type=str,
                    default="1",
                    help="gpu card ID")
parser.add_argument("--meta_weight",
                    type=str,
	                default=False,
	                help="whether to user mete_weight_net")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

torch.manual_seed(2019)  # cpu
torch.cuda.manual_seed(2019)  # gpu
np.random.seed(2019)  # numpy
random.seed(2019)  # random and transforms
torch.backends.cudnn.deterministic = True  # cudnn


def worker_init_fn(worker_id):
    np.random.seed(2019 + worker_id)


data_path = '../data/{}/'.format(args.dataset)
model_path = './models/{}/'.format(args.dataset)
print("arguments: %s " % (args))
print("config model", args.model)
print("config data path", data_path)
print("config model path", model_path)

############################## PREPARE DATASET ##########################

train_data, valid_data, test_data_pos, user_pos, user_num, item_num, train_mat, train_data_noisy = data_utils.load_all(
    args.dataset, data_path)

# construct the train and test datasets
train_dataset = data_utils.NCFData(
    train_data, item_num, train_mat, args.num_ng, 0, train_data_noisy)
valid_dataset = data_utils.NCFData(
    valid_data, item_num, train_mat, args.num_ng, 1)

train_loader = data.DataLoader(train_dataset,
                               batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                               worker_init_fn=worker_init_fn)
valid_loader = data.DataLoader(valid_dataset,
                               batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                               worker_init_fn=worker_init_fn)

print("data loaded! user_num:{}, item_num:{} train_data_len:{} test_user_num:{}".format(user_num, item_num,
                                                                                        len(train_data),
                                                                                        len(test_data_pos)))
########################### CREATE MODEL #################################
if args.model == 'NeuMF-pre':  # pre-training. Not used in our work.
    GMF_model_path = model_path + 'GMF.pth'
    MLP_model_path = model_path + 'MLP.pth'
    NeuMF_model_path = model_path + 'NeuMF.pth'
    assert os.path.exists(GMF_model_path), 'lack of GMF model'
    assert os.path.exists(MLP_model_path), 'lack of MLP model'
    GMF_model = torch.load(GMF_model_path)
    MLP_model = torch.load(MLP_model_path)
else:
    GMF_model = None
    MLP_model = None

model = model.NCF(user_num, item_num, args.factor_num, args.num_layers,
                  args.dropout, args.model, GMF_model, MLP_model)

model.cuda()
BCE_loss = nn.BCEWithLogitsLoss()

if args.model == 'NeuMF-pre':
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


# writer = SummaryWriter() # for visualization
# focal_loss = FocalLoss(gamma = 5)
########################### Eval #####################################
def eval(model, valid_loader, best_loss, count):
    model.eval()
    epoch_loss = 0
    valid_loader.dataset.ng_sample()  # negative sampling
    for user, item, label, noisy_or_not in valid_loader:
        user = user.cuda()
        item = item.cuda()
        label = label.float().cuda()

        prediction = model(user, item)
        loss, weight, _ = loss_function(prediction, label, args.alpha)
        # loss = focal_loss(prediction, label)
        epoch_loss += loss.detach()
    print("################### EVAL ######################")
    print("Eval loss:{}".format(epoch_loss))
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        if args.out:
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            torch.save(model, '{}{}_{}.pth'.format(model_path, args.model, args.alpha))
    return best_loss


########################### Test #####################################
def test(model, test_data_pos, user_pos):
    top_k = args.top_k
    model.eval()
    _, recall, NDCG, _ = evaluate.test_all_users(model, 4096, item_num, test_data_pos, user_pos, top_k)

    print("################### TEST ######################")
    print("Recall {:.4f}-{:.4f}".format(recall[0], recall[1]))
    print("NDCG {:.4f}-{:.4f}".format(NDCG[0], NDCG[1]))


########################### TRAINING #####################################
count, best_hr = 0, 0
best_loss = 1e9

tb_writer = SummaryWriter(args.tb_dir)

for epoch in range(args.epochs):
    model.train()  # Enable dropout (if have).

    start_time = time.time()
    train_loader.dataset.ng_sample()

    for user, item, label, noisy_or_not in train_loader:
        user = user.cuda()
        item = item.cuda()
        label = label.float().cuda()

        if args.meta_weight:
            meta_model = copy.deepcopy(model)
            meta_model.load_state_dict(model.state_dict())
            outputs = meta_model(user, item)
            cost = F.binary_cross_entropy_with_logits(outputs, label, reduction='none')
            cost_v = torch.reshape(cost, (len(cost), 1))
            # cost_2 = F.binary_cross_entropy_with_logits(-outputs, label, reduction='none')
            # cost_v_2 = torch.reshape(cost_2, (len(cost), 1))
            v_lambda = vnet(cost_v.data)
            # l_f_meta = torch.sum(cost_v * v_lambda + cost_v_2 * (1-v_lambda))/len(cost_v)
            l_f_meta = torch.sum(cost_v * v_lambda) / len(cost_v)
            meta_model.zero_grad()
            grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True, allow_unused=True)
            meta_lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 100)))  # For ResNet32
            meta_model.update_params(lr_inner=meta_lr, source_params=grads)
            del grads
            try:
                user_val, item_val, label_val, _ = next(train_meta_loader_iter)
            except StopIteration:
                train_meta_loader_iter = iter(valid_loader)
                user_val, item_val, label_val, _ = next(train_meta_loader_iter)
            user_val, item_val, label_val = user_val.to(args.device), item_val.to(args.device), label_val.float().to(
                args.device)
            y_g_hat = meta_model(user_val, item_val)
            l_g_meta = F.binary_cross_entropy_with_logits(y_g_hat, label_val)
            vnet_optimizer.zero_grad()
            l_g_meta.backward()
            vnet_optimizer.step()

            prediction = model(user, item)
            cost = F.binary_cross_entropy_with_logits(prediction, label, reduction='none')
            cost_w = torch.reshape(cost, (len(cost), 1))
            # cost_2 = F.binary_cross_entropy_with_logits(-prediction, label, reduction='none')
            # cost_w_2 = torch.reshape(cost_2, (len(cost_2), 1))
            with torch.no_grad():
                v_lambda = vnet(cost_w.data)
            model.zero_grad()
            # loss = torch.sum(cost_w * v_lambda + cost_w_2 * (1 - v_lambda)) / len(cost_w)
            loss = torch.sum(cost_w * v_lambda) / len(cost_w)

        else:
            model.zero_grad()
            prediction = model(user, item)
            # loss = F.binary_cross_entropy_with_logits(prediction, label)
            loss, weight = loss_function(prediction, label, args.alpha)

        noisy_index = torch.where(noisy_or_not == True)
        clean_index = torch.where(noisy_or_not == False)
        tb_writer.add_scalar("loss/noisy", torch.mean(loss_batch.reshape(-1)[noisy_index]), count)
        tb_writer.add_scalar("loss/clean", torch.mean(loss_batch.reshape(-1)[clean_index]), count)
        tb_writer.add_scalar("weight/noisy", torch.mean(weight.reshape(-1)[noisy_index]), count)
        tb_writer.add_scalar("weight/clean", torch.mean(weight.reshape(-1)[clean_index]), count)

        if count % args.eval_freq == 0 and count != 0:
            noisy_index = torch.where(noisy_or_not == True)
            clean_index = torch.where(noisy_or_not == False)
            print("noisy weight:", torch.mean(weight.reshape(-1)[noisy_index]))
            print("clean weight:", torch.mean(weight.reshape(-1)[clean_index]))
            print("noisy loss:", torch.mean(loss_batch.reshape(-1)[noisy_index]))
            print("clean loss:", torch.mean(loss_batch.reshape(-1)[clean_index]))
            print("epoch: {}, iter: {}, loss:{}".format(epoch, count, loss))
            best_loss = eval(model, valid_loader, best_loss, count)
            model.train()

        count += 1

print("############################## Training End. ##############################")
test_model = torch.load('{}{}_{}.pth'.format(model_path, args.model, args.alpha))
test_model.cuda()
test(test_model, test_data_pos, user_pos)
