import argparse
import copy
import math
import os.path as osp
import random
from time import perf_counter as t

import numpy as np
import yaml
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv

from loss import BarlowTwinsLoss
import argparse
from mask_generator import FeatureMask
from torch.autograd import Variable
from model import Encoder, MaskModel, drop_feature
from eval import label_classification

import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Meta Mask Training for Simclr on TU datasets')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')

    parser.add_argument('--disable-meta', action="store_true",
                        help='whether using meta learning when training')
    parser.add_argument('--disable-sigmoid',
                        action="store_true",
                        help='whether using meta learning when training')
    parser.add_argument('--no-second-order', action="store_true")
    parser.add_argument('--weight-cl', default=1, type=float,
                        help='weight of simclr loss')
    parser.add_argument('--weight-bl', default=1, type=float,
                        help='weight of simclr loss')
    parser.add_argument('--ckpt-path', type=str, default=None)

    args = parser.parse_args()
    return args


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class MetaMaskBarlowTwins(nn.Module):
    def __init__(self, encoder, num_hidden, num_proj_hidden, tau, device, enable_meta, enable_sigmoid, weight_cl,
                 second_order):
        super(MetaMaskBarlowTwins, self).__init__()
        self.mask_model = MaskModel(encoder, num_hidden, num_proj_hidden, tau)
        self.mask_model_ = copy.deepcopy(self.mask_model)
        self.criterion = BarlowTwinsLoss(device)
        if enable_meta:
            self.auto_mask = FeatureMask(num_hidden, enable_sigmoid)
        self.enable_meta = enable_meta
        self.weight_cl = weight_cl
        self.second_order = second_order

    # x1,x2表示过encoder之后的结果
    def unrolled_backward(self, x1, edge_index_1, x2, edge_index_2, model_optim, mask_optim, eta):
        """
        Compute un-rolled loss and backward its gradients
        """
        #  compute unrolled multi-task network theta_1^+ (virtual step)
        masks = self.auto_mask()

        z1 = self.mask_model(x1, edge_index_1, masks)
        z2 = self.mask_model(x2, edge_index_2, masks)
        p1_bl = model.mask_model.projection_(z1)
        p2_bl = model.mask_model.projection_(z2)
        loss = model.criterion(p1_bl, p2_bl)

        model_optim.zero_grad()
        mask_optim.zero_grad()
        # 计算一步梯度
        loss.backward()
        # copy梯度
        gradients = copy.deepcopy(
            [v.grad.data if v.grad is not None else None for v in self.mask_model.parameters()])

        model_optim.zero_grad()
        mask_optim.zero_grad()
        # do virtual step: theta_1^+ = theta_1 - alpha * (primary loss + auxiliary loss)
        # optimizer.param_groups：是长度为2的list，其中的元素是2个字典；
        # optimizer.param_groups[0]：长度为6的字典，
        # 包括[‘amsgrad’, ‘params’, ‘lr’, ‘betas’, ‘weight_decay’, ‘eps’]这6个参数
        # optimizer.param_groups[1]：好像是表示优化器的状态的一个字典
        with torch.no_grad():
            for weight, weight_, d_p in zip(self.mask_model.parameters(),
                                            self.mask_model_.parameters(),
                                            gradients):
                if d_p is None:
                    weight_.copy_(weight)
                    continue

                d_p = -d_p
                g = model_optim.param_groups[0]
                state = model_optim.state[weight]
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                step_t = state['step']
                step_t += 1

                if g['weight_decay'] != 0:
                    d_p = d_p.add(weight, alpha=g['weight_decay'])
                beta1, beta2 = g['betas']
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(g['betas'][0]).add_(d_p, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(d_p, d_p.conj(), value=1 - beta2)

                # step = step_t.item()
                step = step_t

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = g['lr'] / bias_correction1

                bias_correction2_sqrt = math.sqrt(bias_correction2)
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(g['eps'])

                weight.addcdiv_(exp_avg, denom, value=-step_size)
                weight_ = copy.deepcopy(weight)
                weight_.grad = None

        masks = self.auto_mask()
        z1_ = self.mask_model_(x1, edge_index_1, masks)
        z2_ = self.mask_model_(x2, edge_index_2, masks)
        p1_bl_ = model.mask_model_.projection_(z1_)
        p2_bl_ = model.mask_model_.projection_(z2_)
        loss = model.criterion(p1_bl_, p2_bl_)

        mask_optim.zero_grad()
        loss.backward()

        dalpha = [v.grad for v in self.auto_mask.parameters()]
        if self.second_order:
            vector = [v.grad.data if v.grad is not None else None for v in self.mask_model_.parameters()]

            implicit_grads = self._hessian_vector_product(vector, x1, edge_index_1, x2, edge_index_2)

            for g, ig in zip(dalpha, implicit_grads):
                g.data.sub_(ig.data, alpha=eta)

        for v, g in zip(self.auto_mask.parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _hessian_vector_product(self, gradients, x1, edge_index_1, x2, edge_index_2, r=1e-2):
        with torch.no_grad():
            for weight, weight_ in zip(self.mask_model.parameters(), self.mask_model_.parameters()):
                weight_.copy_(weight)
                weight_.grad = None

        valid_grad = []
        for grad in gradients:
            if grad is not None:
                valid_grad.append(grad)
        R = r / _concat(valid_grad).norm()
        for p, v in zip(self.mask_model_.parameters(), gradients):
            if v is not None:
                p.data.add_(v, alpha=R)

        masks = self.auto_mask()
        z1 = self.mask_model(x1, edge_index_1, masks)
        z2 = self.mask_model(x2, edge_index_2, masks)
        p1_bl = model.mask_model.projection_(z1)
        p2_bl = model.mask_model.projection_(z2)
        loss = model.criterion(p1_bl, p2_bl)

        grads_p = torch.autograd.grad(loss, self.auto_mask.parameters())

        for p, v in zip(self.mask_model_.parameters(), gradients):
            if v is not None:
                p.data.sub_(v, alpha=2 * R)

        masks = self.auto_mask()
        z1 = self.mask_model(x1, edge_index_1, masks)
        z2 = self.mask_model(x2, edge_index_2, masks)
        p1_bl = model.mask_model.projection_(z1)
        p2_bl = model.mask_model.projection_(z2)
        loss = model.criterion(p1_bl, p2_bl)

        grads_n = torch.autograd.grad(loss, self.auto_mask.parameters())

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    def per_optimizer_step(self,
                           optimizer_a=None,
                           optimizer_b=None,
                           loss=None):

        # update params
        if loss is not None:
            optimizer_a.zero_grad()
            if optimizer_b is not None:
                optimizer_b.zero_grad()
            loss.backward()
        if optimizer_a is not None:
            optimizer_a.step()
            optimizer_a.zero_grad()
        if optimizer_b is not None:
            optimizer_b.step()
            optimizer_b.zero_grad()


def atest(model: MaskModel, x, edge_index, y, final=False, masks=None):
    model.eval()
    z = model(x, edge_index, masks)

    label_classification(z, y, ratio=0.1)


def setup_seed(seed):
    # torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    # random.seed(seed)

if __name__ == '__main__':
    args = parse_args()
    setup_seed(0)
    enable_meta = not args.disable_meta
    enable_sigmoid = not args.disable_sigmoid
    second_order = not args.no_second_order
    enable_meta = False
    enable_sigmoid = False
    second_order = False

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    torch.manual_seed(config['seed'])
    random.seed(12345)

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    num_epochs = 500
    weight_decay = config['weight_decay']


    def get_dataset(path, name):
        assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']
        name = 'dblp' if name == 'DBLP' else name

        return (CitationFull if name == 'dblp' else Planetoid)(
            path,
            name,
            transform=T.NormalizeFeatures())


    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    encoder = Encoder(dataset.num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers).to(device)
    model = MetaMaskBarlowTwins(encoder, num_hidden, num_proj_hidden, tau, device, enable_meta, enable_sigmoid,
                                args.weight_cl,
                                second_order).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if enable_meta:
        mask_optim = torch.optim.Adam(model.auto_mask.parameters(), lr=0.01)
        # mask_optim = torch.optim.SGD(model.auto_mask.parameters(), lr=0.001, weight_decay=0.0005)
    # start = t()
    # prev = start
    for epoch in range(1, num_epochs + 1):
        loss_all = 0
        loss_bl_all = 0
        loss_cl_all = 0
        model.train()
        optimizer.zero_grad()
        edge_index_1 = dropout_adj(data.edge_index, p=drop_edge_rate_1)[0]
        edge_index_2 = dropout_adj(data.edge_index, p=drop_edge_rate_2)[0]
        x1 = drop_feature(data.x, drop_feature_rate_1)
        x2 = drop_feature(data.x, drop_feature_rate_2)

        mask = torch.ones(num_hidden).to(device)
        if enable_meta:
            mask = model.auto_mask()
        z1 = model.mask_model(x1, edge_index_1, mask)
        z2 = model.mask_model(x2, edge_index_2, mask)
        loss_cl = model.mask_model.loss(z1, z2, batch_size=0)
        # print("loss_cl: ", loss_cl)
        p1_bl = model.mask_model.projection_(z1)
        p2_bl = model.mask_model.projection_(z2)
        loss_bl = model.criterion(p1_bl, p2_bl)
        loss = args.weight_bl * loss_bl + args.weight_cl * loss_cl
        if enable_meta:
            model.per_optimizer_step(optimizer, None, loss)
            model.unrolled_backward(x1, edge_index_1, x2, edge_index_2, optimizer, mask_optim,
                                    optimizer.param_groups[0]['lr'])
            model.per_optimizer_step(None, mask_optim, None)
        else:
            loss.backward()
            optimizer.step()
        print('Epoch {}, Loss {}, Loss_bl {}, Loss_cl {}'.format(epoch, loss,
                                                                 loss_bl,
                                                                 loss_cl))

        # now = t()
        # print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
        #       f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        # prev = now
        if epoch % 100 == 0:
            model.eval()
            if enable_meta:
                mask = model.auto_mask()
                print(mask)
            atest(model.mask_model, data.x, data.edge_index, data.y, final=False)

    print("=== Final ===")
    atest(model.mask_model, data.x, data.edge_index, data.y, final=True)
