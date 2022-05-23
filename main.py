# %%
import os
import argparse
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import torch
import matplotlib.pyplot as plt
from IPython.display import display
from torch.utils.data import TensorDataset, DataLoader
from tqdm.autonotebook import tqdm

from utils.logger import *
from utils.draw import *
from utils.grammar import *
from utils.symfind import symfind

import satnet
import symsatnet


def run(epoch, model, projector, optimizer, loader, logger, figlogger, save_dir, to_train = False):
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
        torch.save(model.S, os.path.join(save_dir + "/layers", f'{epoch}.pt'))
        print('TRAINING SET RESULTS: Average loss: {:.4f} Err: {:.4f}'.format(loss_total, err_total))

    else:
        print('TESTING SET RESULTS: Average loss: {:.4f} Err: {:.4f}'.format(loss_total, err_total))

    if to_train and projector and not projector.auto and epoch % projector.proj_period == 0:
        S = model.S.detach().cpu()
        new_S = projector.proj_S(S, projector.proj_lr)
        new_S.requires_grad = True
        model.S = torch.nn.Parameter(new_S.cuda())
        print('S is Projected: {}'.format(projector))
        with open(save_dir + "/grammar.csv", "a") as f:
            f.write(f"{epoch},{projector}\n")
            f.close()
        
    end.record()
    torch.cuda.synchronize()

    logger.log((epoch, loss_total, err_total, start.elapsed_time(end)))
    figlogger.log((epoch, loss_total, err_total))

    
    #print('memory: {:.2f} MB, cached: {:.2f} MB'.format(torch.cuda.memory_allocated()/2.**20, torch.cuda.memory_cached()/2.**20))
    torch.cuda.empty_cache()


def train(epoch, model, projector, optimizer, loader, logger, figlogger, save_dir):
    run(epoch, model, projector, optimizer, loader, logger, figlogger, save_dir, True)


@torch.no_grad()
def test(epoch, model, projector, optimizer, loader, logger, figlogger, save_dir):
    run(epoch, model, projector, optimizer, loader, logger, figlogger, save_dir, False)


def trial(problem, model, trial_num, gpu_num):
    print("===> Trial: {}".format(trial_num))
    print("===> Problem: {}".format(problem))
    print("===> Model: {}".format(model))
    print("===> Setting hyperparameters")

    n = {"sudoku": 729, "cube": 324}[problem]
    data_dir = {"sudoku": "dataset/sudoku_10000", "cube": "dataset/cube_10000"}[problem]
    save_dir = problem + f"_trial_{trial_num}/" + model

    rank = {"SATNet-Plain": 1, "SATNet-300aux": 1, "SymSATNet": 1, "SymSATNet-Auto": 1}[model]
    aux = {"SATNet-Plain": 0, "SATNet-300aux": 300, "SymSATNet": 0, "SymSATNet-Auto": 0}[model]
    lr = {"SATNet-Plain": 2e-3, "SATNet-300aux": 2e-3, "SymSATNet": 4e-2, "SymSATNet-Auto": 2e-3}[model]

    if model == "Proj-Mild":
        projector = {"sudoku": Sudoku, "cube": Cube}[problem]().symmetrize()
        projector.proj_period = {"sudoku": 5, "cube": 7}[problem]
        projector.proj_lr = {"sudoku": 0.2, "cube": 0.2}[problem]
        projector.auto = False
    elif model == "Proj-Sharp":
        projector = {"sudoku": Sudoku, "cube": Cube}[problem]().symmetrize()
        projector.proj_period = {"sudoku": 10, "cube": 10}[problem]
        projector.proj_lr = {"sudoku": 1, "cube": 1}[problem]
        projector.auto = False
    elif model == "SymSATNet":
        projector = {"sudoku": Sudoku, "cube": Cube}[problem]().symmetrize()
        projector.proj_period = {"sudoku": float("inf"), "cube": float("inf")}[problem]
        projector.proj_lr = {"sudoku": 0, "cube": 0}[problem]
        projector.auto = False
    elif model == "SymSATNet-Auto":
        projector = Id(dim = n)
        projector.proj_period = {"sudoku": 10, "cube": 20}[problem]
        projector.proj_lr = {"sudoku": 1, "cube": 1}[problem]
        projector.rtol = {"sudoku": 5e-2, "cube": 1e-1}[problem]
        projector.auto = True
    else:
        projector = None

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
    nTrain = int(N * (1 - testPct))
    nTest = N - nTrain
    assert(nTrain % batchSz == 0)
    assert(nTest % testBatchSz == 0)

    is_input = X.sum(dim = 3, keepdim = True).expand_as(X).int().sign()
    X, Y, is_input = X.cuda(), Y.cuda(), is_input.cuda()

    train_set = TensorDataset(X[:nTrain], is_input[:nTrain], Y[:nTrain])
    test_set =  TensorDataset(X[nTrain:], is_input[nTrain:], Y[nTrain:])

    print('===> Building models')

    if model == "SymSATNet":
        sat = symsatnet.SymSATNet(n, projector.basis()).cuda()
    else:
        sat = satnet.SATNet(n, int(n * rank) + 1, aux).cuda()

    optimizer = torch.optim.Adam(sat.parameters(), lr=lr)

    print('===> Loading loggers')

    if os.path.isdir(save_dir):
        print(f"Directory already exists: {save_dir}")
        return
    os.makedirs(save_dir)
    os.makedirs(save_dir + "/logs")
    os.makedirs(save_dir + "/layers")

    train_logger = CSVLogger(os.path.join(save_dir, 'logs/train.csv'))
    test_logger = CSVLogger(os.path.join(save_dir, 'logs/test.csv'))
    fields = ['epoch', 'loss', 'err', 'time']
    train_logger.log(fields)
    test_logger.log(fields)

    plt.ioff()
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    plt.subplots_adjust(wspace=0.4)
    figtrain_logger = FigLogger(fig, axes[0], 'Traininig')
    figtest_logger = FigLogger(fig, axes[1], 'Testing')
    
    for epoch in range(1, nEpoch+1):
        if model == "SymSATNet-Auto" and projector.auto and projector.proj_period + 1 == epoch:
            S = sat.S.detach().cpu()
            C = S @ S.T
            projector = symfind(C[1:, 1:], rtol = projector.rtol)
            basis = projector.basis()
            upper = projector.proj(C[0, 1:], projector.orbits())
            coeff = coordinates(C[1:, 1:], basis)

            sat = symsatnet.SymSATNet(n, basis).cuda()
            sat.coeff = torch.nn.Parameter(coeff.cuda())
            sat.upper = torch.nn.Parameter(upper.cuda())

            print('S is Projected: {}'.format(projector))
            with open(save_dir + "/grammar.csv", "a") as f:
                f.write(f"{epoch},{projector}\n")
                f.close()

            projector.proj_period = float("inf")
            projector.proj_lr = 0
            projector.auto = True

        train_loader = tqdm(DataLoader(train_set, batch_size = batchSz))
        train(epoch, sat, projector, optimizer, train_loader, train_logger, figtrain_logger, save_dir)
        test_loader = tqdm(DataLoader(test_set, batch_size = testBatchSz))
        test(epoch, sat, projector, optimizer, test_loader, test_logger, figtest_logger, save_dir)
        display(fig)


def main(trial_num, problem_num, model_num, gpu_num):
    problems = ["sudoku", "cube"]
    models = ["SATNet-Plain", "SATNet-300aux", "SymSATNet", "SymSATNet-Auto"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_num', type=int, default=trial_num)
    parser.add_argument('--problem', type=str, default=problems[problem_num])
    parser.add_argument('--model', type=str, default=models[model_num])
    parser.add_argument('--gpu_num', type=int, default=gpu_num)
    args = parser.parse_args(args = [])
    
    assert args.problem in problems
    assert args.model in models

    trial(args.problem, args.model, args.trial_num, args.gpu_num)


# if __name__== '__main__':
#     for problem_num in [0]:
#         for trial_num in [3, 6]:
#             for model_num in [7, 6, 1, 3]:
#                 main(trial_num = trial_num, problem_num = problem_num, model_num = model_num, gpu_num = 0)


# if __name__ == '__main__':
#     import time
#     G = Kron(Sum(Wreath(Perm(2), Perm(3)), Sum(Wreath(Perm(3), Perm(8)), Kron(Perm(2), Id(10)))), Perm(6))
#     # G = Cube()
#     # G = Kron(Perm(30), Id(10))
#     # G = Kron(Id(15), Id(15))
#     C = torch.rand(G.dim, G.dim)

#     print(G.n_basis())

#     start = time.time()
#     G.proj(C)
#     print(time.time() - start)

#     start = time.time()
#     G._proj(C)
#     print(time.time() - start)

# if __name__ == '__main__':
#     with open("results/validation_results/cube_trial_3_corrupt_0/SymSATNet-Val/layers/20.pt", "rb") as f:
#     with open("results/corrupt_results/sudoku_trial_1_corrupt_3/SymSATNet-Val/layers/20.pt", "rb") as f:
#         C = torch.load(f).detach().cpu()[1:, 1:]
#         C = C @ C.T

#     draw(C)
#     print(symfind(C, 0.065))
# %%
