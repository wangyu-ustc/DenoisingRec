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
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

from model import NCF, Gamma
import evaluate
import data_utils
from loss import loss_function

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    type=str,
                    help='dataset used for training, options: amazon_book, yelp, adressa',
                    default='adressa')
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
parser.add_argument("--pretrain_epochs",
                    type=int,
                    default=1,
                    help="pretrain the model")
parser.add_argument("--eval_freq",
                    type=int,
                    default=2000,
                    help="the freq of eval")
parser.add_argument("--top_k",
                    type=list,
                    default=[3, 20],
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
parser.add_argument("--device",
                    default='cuda',
                    help='cuda or cpu')
# parser.add_argument("--gpu",
#                     type=str,
#                     default="1",
#                     help="gpu card ID")
parser.add_argument("--meta_weight",
                    type=str,
	                default=False,
	                help="whether to user mete_weight_net")
parser.add_argument("--emb_dim",
                    type=int,
                    default=32,
                    help='embedding dimension of the Gamma model')
parser.add_argument("--use_VAE",
                    type=int,
                    default=True,
                    help='whether to use VAE-based training')
parser.add_argument("--lambda_1",
                    type=float,
                    default=0.01,
                    help='regularization')
parser.add_argument("--lambda_2",
                    type=float,
                    default=0.01,
                    help='regularization')
parser.add_argument("--prior",
                    type=str,
                    default='pretrain',
                    help='priors used for experiments, options: MF, pretrain')
parser.add_argument("--training_method",
                    type=str,
                    default='iterative',
                    help="training routine, options: normal, iterative")
args = parser.parse_args()
cudnn.benchmark = True

torch.manual_seed(2019)  # cpu
torch.cuda.manual_seed(2019)  # gpu
np.random.seed(2019)  # numpy
random.seed(2019)  # random and transforms
torch.backends.cudnn.deterministic = True  # cudnn
CONSTANT = 1e-5

def worker_init_fn(worker_id):
    np.random.seed(2019 + worker_id)


data_path = '../data/{}/'.format(args.dataset)
model_path = './models/{}/'.format(args.dataset)
print("arguments: %s " % (args))
print("config model", args.model)
print("config data path", data_path)
print("config model path", model_path)

############################## PREPARE DATASET ##########################

train_data, valid_data, test_data_pos, test_data_noisy, user_pos, user_num, item_num, train_mat, train_data_noisy = data_utils.load_all(
    args.dataset, data_path)

# construct the train and test datasets
train_dataset = data_utils.NCFData(
    train_data, item_num, train_mat, 0, 0, train_data_noisy)
valid_dataset = data_utils.NCFData(
    valid_data, item_num, train_mat, args.num_ng, 1)


train_loader = data.DataLoader(train_dataset,
                               batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,
                               worker_init_fn=worker_init_fn)
valid_loader = data.DataLoader(valid_dataset,
                               batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,
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

model = NCF(user_num, item_num, args.factor_num, args.num_layers,
                  args.dropout, args.model, GMF_model, MLP_model)

model.to(args.device)
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
        user = user.to(args.device)
        item = item.to(args.device)
        label = label.float().to(args.device)

        prediction = model(user, item)
        # loss = loss_function(prediction, label, args.alpha)
        loss = F.binary_cross_entropy(prediction, label)
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

Gamma_1 = Gamma(user_num=user_num, item_num=item_num, K0=args.emb_dim).to(args.device)
if args.prior == 'MF':
    Gamma_2 = Gamma(user_num=user_num, item_num=item_num, K0=args.emb_dim).to(args.device)
else:
    Gamma_2 = copy.deepcopy(model)

gamma_1_optim = optim.Adam(Gamma_1.parameters(), lr=args.lr)
gamma_2_optim = optim.Adam(Gamma_2.parameters(), lr=args.lr)


train_loader.dataset.ng_sample() # 形式化处理，不会真的去获取netative samples

if args.prior == 'pretrain':
    print("Baseline: pretrain model...")
    for epoch in range(args.pretrain_epochs):
        Gamma_2.train()
        for user, item, label, noisy_or_not in train_loader:
            user = user.to(args.device)
            item = item.to(args.device)
            label = label.float().to(args.device)
            Gamma_2.zero_grad()
            prediction = Gamma_2(user, item)
            loss = F.binary_cross_entropy(prediction, label)
            # loss = loss_function(prediction, label, args.alpha)

            for _ in range(args.num_ng):
                neg_item = []
                for single_user in user:
                    j = np.random.randint(item_num)
                    while (single_user, j) in train_mat:
                        j = np.random.randint(item_num)
                    neg_item.append(j)
                neg_item = torch.tensor(neg_item).to(args.device)
                neg_prediction = Gamma_2(user, neg_item)
                loss += F.binary_cross_entropy(neg_prediction, torch.zeros_like(label))
                # loss += loss_function(prediction, label, args.alpha)
            loss.backward()
            gamma_2_optim.step()
            if count % 200 == 0 and count != 0:
                print("epoch: {}, iter: {}, loss:{}".format(epoch, count, loss))

            if count % args.eval_freq == 0 and count != 0:
                test(Gamma_2, test_data_pos, user_pos)
                best_loss = eval(Gamma_2, valid_loader, best_loss, count)
                Gamma_2.train()
            count += 1
    best_loss = eval(Gamma_2, valid_loader, best_loss, count)
    test(Gamma_2, test_data_pos, user_pos)
    Gamma_2.eval()


for epoch in range(args.epochs):
    model.train()  # Enable dropout (if have).

    start_time = time.time()

    if args.training_method == 'normal':
        for user, item, label, noisy_or_not in train_loader:
            user = user.to(args.device)
            item = item.to(args.device)
            label = label.float().to(args.device)
            model.zero_grad()
            prediction = model(user, item)
            if args.use_VAE == True:
                loss_1 = torch.log(Gamma_1(user, item) + CONSTANT) * (1 - prediction)
                if args.prior == 'pretrain':
                    p = Gamma_2(user, item).detach()
                else:
                    p = Gamma_2(user, item)
                # loss_2 = p * torch.log(CONSTANT + p) - p * torch.log(CONSTANT + prediction) \
                #          + (1 - p) * torch.log(CONSTANT + 1 - p) - (1 - p) * torch.log(CONSTANT + 1 - prediction)
                loss_2 = prediction * torch.log(CONSTANT + prediction) - prediction * torch.log(CONSTANT + p) \
                         + (1 - prediction) * torch.log(CONSTANT + 1 - prediction) - (1 - prediction) * torch.log(CONSTANT + 1 - p)
                loss = loss_2 - loss_1
                for _ in range(args.num_ng):
                    neg_item = []
                    for single_user in user:
                        j = np.random.randint(item_num)
                        while (single_user, j) in train_mat:
                            j = np.random.randint(item_num)
                        neg_item.append(j)
                    neg_item = torch.tensor(neg_item).to(args.device)
                    neg_prediction = model(user, neg_item)
                    neg_loss_1 = -torch.log(1 - Gamma_1(user, neg_item)+ CONSTANT) * (1-neg_prediction) + neg_prediction * 1000
                    # neg_loss_2 = -torch.log(1 - neg_prediction + CONSTANT)
                    if args.prior == 'pretrain':
                        p = Gamma_2(user, neg_item).detach()
                    else:
                        p = Gamma_2(user, neg_item)
                    # neg_loss_2 = (p * torch.log(CONSTANT + p) - p * torch.log(CONSTANT + neg_prediction)
                    # + (1 - p) * torch.log(CONSTANT + 1 - p) - (1 - p) * torch.log(CONSTANT + 1 - neg_prediction))
                    neg_loss_2 = neg_prediction * torch.log(CONSTANT + neg_prediction) - neg_prediction * torch.log(CONSTANT + p) \
                             + (1 - neg_prediction) * torch.log(CONSTANT + 1 - neg_prediction) - (1 - neg_prediction) * torch.log(
                        CONSTANT + 1 - p)

                    loss += (neg_loss_1 + neg_loss_2)

                loss = torch.mean(loss)

                # loss += args.lambda_1 * torch.norm(Gamma_1.user_embedding.weight) + torch.norm(Gamma_1.item_embedding.weight)
                # loss += args.lambda_2 * torch.norm(Gamma_2.user_embedding.weight) + torch.norm(Gamma_2.item_embedding.weight)

                model.zero_grad()
                gamma_1_optim.zero_grad()
                gamma_2_optim.zero_grad()

                loss.backward()

                optimizer.step()
                gamma_1_optim.step()
                gamma_2_optim.step()

            else:
                loss = F.binary_cross_entropy(prediction, label)
                # loss = loss_function(prediction, label, args.alpha)
                for _ in range(args.num_ng):
                    neg_item = []
                    for single_user in user:
                        j = np.random.randint(item_num)
                        while (single_user, j) in train_mat:
                            j = np.random.randint(item_num)
                        neg_item.append(j)
                    neg_item = torch.tensor(neg_item).to(args.device)
                    neg_prediction = model(user, neg_item)
                    loss += F.binary_cross_entropy(neg_prediction, torch.zeros_like(label))
                    # loss += loss_function(prediction, label, args.alpha)
                loss.backward()
                optimizer.step()

            if count % 200 == 0 and count != 0:
                print("epoch: {}, iter: {}, loss:{}".format(epoch, count, loss))

            if count % args.eval_freq == 0 and count != 0:
                test(model, test_data_pos, user_pos)
                best_loss = eval(model, valid_loader, best_loss, count)
                model.train()

            count += 1
    else:
        assert args.use_VAE == True
        assert args.training_method == 'iterative'
        assert args.prior == 'pretrain'
        iterative_step = 1
        for user, item, label, noisy_or_not in train_loader:
            user = user.to(args.device)
            item = item.to(args.device)
            label = label.float().to(args.device)
            # model.zero_grad()
            prediction = model(user, item)

            if count % (iterative_step + 1) == 0:
                # train the decoder, P(\tilde{R}|R), or Gamma_{ui} in the paper
                prediction = prediction.detach()
                process = lambda x: x
            else:
                process = lambda x: x.detach()
            loss_1 = torch.log(process(Gamma_1(user, item)) + CONSTANT) * (1 - prediction)

            p = Gamma_2(user, item).detach()
            loss_2 = prediction * torch.log(CONSTANT + prediction) - prediction * torch.log(CONSTANT + p) \
                     + (1 - prediction) * torch.log(CONSTANT + 1 - prediction) - (1 - prediction) * torch.log(
                CONSTANT + 1 - p)
            loss = loss_2 - loss_1
            for _ in range(args.num_ng):
                neg_item = []
                for single_user in user:
                    j = np.random.randint(item_num)
                    while (single_user, j) in train_mat:
                        j = np.random.randint(item_num)
                    neg_item.append(j)
                neg_item = torch.tensor(neg_item).to(args.device)
                neg_prediction = model(user, neg_item)
                neg_loss_1 = -torch.log(1 - process(Gamma_1(user, neg_item)) + CONSTANT) * (1 - neg_prediction) + \
                                 neg_prediction * 1000
                # neg_loss_2 = -torch.log(1 - neg_prediction + CONSTANT)
                p = Gamma_2(user, neg_item).detach()
                neg_loss_2 = prediction * torch.log(CONSTANT + prediction) - prediction * torch.log(CONSTANT + p) \
                             + (1 - prediction) * torch.log(CONSTANT + 1 - prediction) - (1 - prediction) * torch.log(
                    CONSTANT + 1 - p)

                loss += (neg_loss_1 + neg_loss_2)

            loss = torch.mean(loss)
            model.zero_grad()
            gamma_1_optim.zero_grad()
            gamma_2_optim.zero_grad()

            loss.backward()

            optimizer.step()
            gamma_1_optim.step()
            gamma_2_optim.step()

            count += 1

            if count % 200 == 0 and count != 0:
                print("epoch: {}, iter: {}, loss:{}".format(epoch, count, loss))

            if count % args.eval_freq == 0 and count != 0:
                test(model, test_data_pos, user_pos)
                best_loss = eval(model, valid_loader, best_loss, count)
                model.train()


print("############################## Training End. ##############################")
# test_model = torch.load('{}{}_{}.pth'.format(model_path, args.model, args.alpha))
# test_model.to(args.device)
print("Clean Test:")
test(model, test_data_pos, user_pos)

print("Noisy Test:")
test(model, test_data_noisy, user_pos)

if args.use_VAE:
    torch.save(Gamma_1, '{}{}.pth'.format(model_path, "Gamma_1"))
    torch.save(Gamma_2, '{}{}.pth'.format(model_path, "Gamma_2"))

    Gamma = -np.ones([100, 100])
    Gamma_backward = -np.ones([100, 100])
    Gamma_1.to("cpu")
    Gamma_2.to("cpu")
    for u in range(100):
        for i in range(100):
            if (u,i) in train_mat:
                Gamma[u,i] = Gamma_1(torch.tensor(u).unsqueeze(0),torch.tensor(i).unsqueeze(0))
                Gamma_backward[u,i] = Gamma_2(torch.tensor(u).unsqueeze(0),torch.tensor(i).unsqueeze(0))

    print("Gamma")
    print(Gamma.round(decimals=4))
    print("Gamma_backward")
    print(Gamma_backward.round(decimals=4))
    # import pandas as pd
    # pd.DataFrame(Gamma.round(decimals=4)).to_csv("C:/Users/ls/Desktop/Gamma_1.csv",header=False, index=False, sep=',')
    # pd.DataFrame(Gamma_backward.round(decimals=4)).to_csv("C:/Users/ls/Desktop/Gamma_2.csv",header=False, index=False, sep=',')




