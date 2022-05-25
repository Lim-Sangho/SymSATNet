# %%
from __future__ import annotations
import os
import argparse
import warnings
from typing import Callable
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

import torch
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from torch.utils.data import TensorDataset, DataLoader
from tqdm.autonotebook import tqdm
from time import sleep

from utils.logger import *
from utils.draw import *
from utils.grammar import *
from utils.group import *
from utils.symfind import *

import satnet
import symsatnet


def run(epoch, model, group, optimizer, loader, logger, figlogger, timelogger, save, save_dir, to_train = False):
    loss_total, err_total = 0, 0
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)

    start.record()
    
    for data, is_input, label in loader:
        if to_train:
            optimizer.zero_grad()

        preds = model(data.flatten(1).contiguous(), is_input.flatten(1).contiguous()).reshape(data.shape)
        loss = torch.nn.BCELoss()(preds, label)

        if to_train:
            loss.backward()
            optimizer.step()

        preds = preds.argmax(3).flatten(1)
        label = label.argmax(3).flatten(1)
        err = 1 - torch.sum(torch.all(preds == label, dim = 1)) / preds.size(0)

        loader.set_description('Epoch {} {} Loss {:.4f} Err: {:.4f}'.format(epoch, ('Train' if to_train else 'Test '), loss.item(), err.item()))
        loss_total += loss.item()
        err_total += err.item()

    loss_total, err_total = loss_total/len(loader), err_total/len(loader)

    if to_train:
        if save:
            torch.save(model.S, os.path.join(save_dir + "/layers", f'{epoch}.pt'))
        print('TRAINING SET RESULTS: Average loss: {:.4f} Err: {:.4f}'.format(loss_total, err_total))

    else:
        print('TESTING SET RESULTS: Average loss: {:.4f} Err: {:.4f}'.format(loss_total, err_total))
        
    end.record()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    if save:
        if logger is not None:
            logger.log([epoch, loss_total, err_total, start.elapsed_time(end) / 1000])
    if figlogger is not None:
        figlogger.log([epoch, loss_total, err_total])
    if timelogger is not None:
        timelogger.log(start.elapsed_time(end) / 1000)

    return err_total


def train(epoch, model, group, optimizer, loader, logger, figlogger, timelogger, save, save_dir):
    return run(epoch, model, group, optimizer, loader, logger, figlogger, timelogger, save, save_dir, True)


@torch.no_grad()
def test(epoch, model, group, optimizer, loader, logger, figlogger, timelogger, save, save_dir):
    return run(epoch, model, group, optimizer, loader, logger, figlogger, timelogger, save, save_dir, False)


@torch.no_grad()
def validation(grammar: Grammar, construct_group: Callable[[Grammar], Group], S: torch.Tensor, valid_err: float, valid_args: list, eps: float) -> Grammar:
    print(grammar)

    new_group = construct_group(grammar)
    new_S = new_group.proj_S(S)
    new_S.requires_grad = True
    valid_args[1].S = torch.nn.Parameter(new_S.cuda())
    err = test(*valid_args)
    if err <= valid_err - eps:
        return grammar

    if isinstance(grammar, Id) or isinstance(grammar, Cyclic) or isinstance(grammar, Symm):
        return Id(dim = grammar.dim)

    if isinstance(grammar, Kron):
        construct_1 = lambda G: construct_group(Kron(G, Id(dim = grammar.G2.dim)))
        construct_2 = lambda G: construct_group(Kron(Id(dim = grammar.G1.dim), G))
        new_G1 = validation(grammar.G1, construct_1, S, valid_err, valid_args, eps)
        new_G2 = validation(grammar.G2, construct_2, S, valid_err, valid_args, eps)
        return Kron(new_G1, new_G2)

    if isinstance(grammar, Sum):
        construct_1 = lambda G: construct_group(Sum(G, Id(dim = grammar.G2.dim)))
        construct_2 = lambda G: construct_group(Sum(Id(dim = grammar.G1.dim), G))
        new_G1 = validation(grammar.G1, construct_1, S, valid_err, valid_args, eps)
        new_G2 = validation(grammar.G2, construct_2, S, valid_err, valid_args, eps)
        return Sum(new_G1, new_G2)

    if isinstance(grammar, Wreath):
        construct_1 = lambda G: construct_group(Wreath(G, Id(dim = grammar.G2.dim)))
        construct_2 = lambda G: construct_group(Wreath(Id(dim = grammar.G1.dim), G))
        new_G1 = validation(grammar.G1, construct_1, S, valid_err, valid_args, eps)
        new_G2 = validation(grammar.G2, construct_2, S, valid_err, valid_args, eps)
        return Wreath(new_G1, new_G2)


def main(trial_num = 1, problem = "sudoku", model = "SymSATNet", corrupt_num = 0, gpu_num = 0, save = False):
    print("===> Trial: {}".format(trial_num))
    print("===> Problem: {}".format(problem))
    print("===> Model: {}".format(model))
    print("===> Setting hyperparameters")

    n = {"sudoku": 729, "cube": 324}[problem]
    data_dir = {"sudoku": "dataset/sudoku_10000", "cube": "dataset/cube_10000"}[problem]
    save_dir = ".results/" + problem + f"_trial_{trial_num}_corrupt_{corrupt_num}/" + model

    rank = {"SATNet-Plain": 1, "SATNet-300aux": 1, "SymSATNet": 1, "SymSATNet-Auto": 1}[model]
    aux = {"SATNet-Plain": 0, "SATNet-300aux": 300, "SymSATNet": 0, "SymSATNet-Auto": 0}[model]
    lr = {"SATNet-Plain": 2e-3, "SATNet-300aux": 2e-3, "SymSATNet": 4e-2, "SymSATNet-Auto": 2e-3}[model]

    if model == "SymSATNet":
        grammar = {"sudoku": Sudoku, "cube": Cube}[problem]().symmetrize()
        perm = torch.arange(n)
        proj_period = {"sudoku": float("inf"), "cube": float("inf")}[problem]
        proj_lr = {"sudoku": 0.0, "cube": 0.0}[problem]
        group = Group(grammar, perm, proj_period, proj_lr)
    elif model == "SymSATNet-Auto":
        grammar = Id(dim = n)
        perm = torch.arange(n)
        proj_period = {"sudoku": 10, "cube": 20}[problem]
        proj_lr = {"sudoku": 1.0, "cube": 1.0}[problem]
        rtol = {"sudoku": [0.05, 0.05, 0.055, 0.06], "cube": [0.1, 0.1, 0.11, 0.12]}[problem][corrupt_num]
        group = Group(grammar, perm, proj_period, proj_lr, rtol)
        eps = [0.0, 0.0, 0.1, 0.2][corrupt_num]
    else:
        group = None

    batchSz, testBatchSz, testPct, nEpoch = 40, 40, 0.1, 100

    print('===> Initializing CUDA')

    assert torch.cuda.is_available()
    torch.cuda.set_device(torch.device("cuda:{}".format(gpu_num)))
    print('===> Using', torch.cuda.get_device_name(gpu_num))
    print('===> Using', "cuda:{}".format(gpu_num))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.init()

    print('===> Loading dataset')

    with open(os.path.join(data_dir, 'features.pt'), 'rb') as f:
        X = torch.load(f)
    with open(os.path.join(data_dir, 'labels.pt'), 'rb') as f:
        Y = torch.load(f)

    N = X.size(0)
    batchSz, testBatchSz, validBatchSz = 40, 40, 40
    testPct, validPct = (0.1, 0.1) if model == "SymSATNet-Auto" else (0.1, 0.0)
    nTrain = int(N * (1 - testPct - validPct))
    nTest = int(N * testPct)
    nValid = int(N * validPct)
    assert(nTrain % batchSz == 0)
    assert(nTest % testBatchSz == 0)
    assert(nValid % validBatchSz == 0)

    # Noisy train dataset
    for feature, label in zip(X[:nTrain], Y[:nTrain]):
        p, q, r = label.shape
        indices = np.random.choice(range(p * q), corrupt_num, replace = False)
        indices = [(index // q, index % q) for index in indices]

        for pair in indices:
            entry = torch.argmax(label[pair])
            corrupt = torch.randint(0, r, (1,))
            while corrupt == entry:
                corrupt = torch.randint(0, r, (1,))
            label[pair] = torch.nn.functional.one_hot(corrupt, r)
            if not torch.all(feature[pair] == 0):
                feature[pair] = torch.nn.functional.one_hot(corrupt, r)

    is_input = X.sum(dim = 3, keepdim = True).expand_as(X).int().sign()
    X, Y, is_input = X.cuda(), Y.cuda(), is_input.cuda()

    train_set = TensorDataset(X[:nTrain], is_input[:nTrain], Y[:nTrain])
    test_set = TensorDataset(X[nTrain:nTrain+nTest], is_input[nTrain:nTrain+nTest], Y[nTrain:nTrain+nTest])
    if model == "SymSATNet-Auto":
        valid_set = TensorDataset(X[nTrain+nTest:], is_input[nTrain+nTest:], Y[nTrain+nTest:])

    print('===> Building models')

    if model == "SymSATNet":
        if isinstance(grammar, Id) or grammar.n_basis() > 500:
            sat = symsatnet.SymSATNet_group(n, group).cuda()
        else:
            sat = symsatnet.SymSATNet_basis(n, grammar.basis()).cuda()
    else:
        sat = satnet.SATNet(n, int(n * rank) + 1, aux).cuda()

    optimizer = torch.optim.Adam(sat.parameters(), lr=lr)

    if save:
        print('===> Loading loggers')

        if os.path.isdir(save_dir):
            print(f"Directory already exists: {save_dir}")
            return
        os.makedirs(save_dir)
        os.makedirs(save_dir + "/logs")
        os.makedirs(save_dir + "/layers")

        train_logger = CSVLogger(os.path.join(save_dir, 'logs/train.csv'))
        test_logger = CSVLogger(os.path.join(save_dir, 'logs/test.csv'))
        valid_logger = CSVLogger(os.path.join(save_dir, 'logs/validation.csv'))
        train_logger.log(['epoch', 'loss', 'err', 'time'])
        test_logger.log(['epoch', 'loss', 'err', 'time'])
        valid_logger.log(['epoch', 'group', 'time'])
    
    else:
        train_logger = None
        test_logger = None
        valid_logger = None

    plt.ioff()
    fig, axes = plt.subplots(1,3, figsize=(15,4))
    plt.subplots_adjust(wspace=0.4)
    figtrain_logger = FigLogger(fig, axes[0], 'Traininig')
    figtest_logger = FigLogger(fig, axes[1], 'Test')
    figtime_logger = TimeLogger(fig, axes[2], 'Time')

    for epoch in range(1, nEpoch+1):
        if model == "SymSATNet-Auto" and epoch == group.proj_period + 1:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing = True)
            end = torch.cuda.Event(enable_timing = True)
            start.record()

            # Find a group with an automatic detection algorithm
            S = sat.S.detach().cpu()
            C = S @ S.T
            grammar, perm = symfind(C[1:, 1:], rtol = group.rtol)
            group = group.set_grammar(grammar).set_perm(perm)

            # Find a useful subgroup with a validation step
            valid_loader = tqdm(DataLoader(valid_set, batch_size = validBatchSz))
            valid_args = [epoch, sat, group, None, valid_loader, None, None, None, save, save_dir]
            valid_err = test(*valid_args)
            grammar = validation(group.grammar, group.set_grammar, S, valid_err, valid_args, eps)
            group = group.set_grammar(grammar)

            if isinstance(group.grammar, Id) or group.grammar.n_basis() > 500:
                sat = symsatnet.SymSATNet_group(n, group).cuda()
                coeff = group._backward(C[1:, 1:].cuda(), 'coeff')
            else:
                basis = group.grammar.basis().cuda()
                basis = basis[:,group.perm][:,:,group.perm]
                sat = symsatnet.SymSATNet_basis(n, basis).cuda()
                coeff = coordinates(C[1:, 1:].cuda(), basis)

            upper = group.cuda().proj_orbit(C[0, 1:].cuda())
            sat.coeff = torch.nn.Parameter(coeff.cuda())
            sat.upper = torch.nn.Parameter(upper.cuda())

            end.record()
            torch.cuda.synchronize()

            print('S is Projected: {}'.format(group))
            print("Elapsed time: {}".format(start.elapsed_time(end)))
            if save:
                valid_logger.log([epoch, group, start.elapsed_time(end)])

            group.proj_period = float("inf")
            group.proj_lr = 0

        train_loader = tqdm(DataLoader(train_set, batch_size = batchSz))
        train(epoch, sat, group, optimizer, train_loader, train_logger, figtrain_logger, figtime_logger, save, save_dir)
        test_loader = tqdm(DataLoader(test_set, batch_size = testBatchSz))
        test(epoch, sat, group, optimizer, test_loader, test_logger, figtest_logger, None, save, save_dir)
        clear_output(wait = True)
        display(fig)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_num', type=int, default=1)
    parser.add_argument('--problem', type=str, default='sudoku')
    parser.add_argument('--model', type=str, default='SymSATNet')
    parser.add_argument('--corrupt_num', type=int, default=0)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--save', action="store_true", default=False)
    args = parser.parse_args()

    assert args.problem in ["sudoku", "cube"]
    assert args.model in ["SATNet-Plain", "SATNet-300aux", "SymSATNet", "SymSATNet-Auto"]

    main(args.trial_num, args.problem, args.model, args.corrupt_num, args.gpu_num, args.save)
# %%
