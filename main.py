# %%
from functools import reduce
import os
import argparse
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from torch.utils.data import TensorDataset, DataLoader
from tqdm.autonotebook import tqdm

from utils.logger import *
from utils.draw import *
from utils.grammar import *
from utils import symfind
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


def trial(problem, model, trial_num):
    print("===> Trial: {}".format(trial_num))
    print("===> Problem: {}".format(problem))
    print("===> Model: {}".format(model))
    print("===> Setting hyperparameters")

    n = {"sudoku": 729, "cube": 324}[problem]
    data_dir = {"sudoku": "dataset/sudoku_10000", "cube": "dataset/cube_10000_2_2_1"}[problem]
    save_dir = problem + f"_trial_{trial_num}/" + model

    rank = {"SATNet-Low": 5/6, "SATNet-Full": 1, "SATNet-Low-300aux": 5/6, "SATNet-Full-300aux": 1,
            "Proj-Mild": 1, "Proj-Sharp": 1, "SymSATNet": 1, "AutoSymSATNet": 1}[model]
    aux = {"SATNet-Low": 0, "SATNet-Full": 0, "SATNet-Low-300aux": 300, "SATNet-Full-300aux": 300,
           "Proj-Mild": 0, "Proj-Sharp": 0, "SymSATNet": 0, "AutoSymSATNet": 0}[model]
    lr = {"SATNet-Low": 2e-3, "SATNet-Full": 2e-3, "SATNet-Low-300aux": 2e-3, "SATNet-Full-300aux": 2e-3,
          "Proj-Mild": 2e-3, "Proj-Sharp": 2e-3, "SymSATNet": 4e-2, "AutoSymSATNet": 2e-3}[model]

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
    elif model == "AutoSymSATNet":
        projector = Id(dim = n)
        projector.proj_period = {"sudoku": 10, "cube": 20}[problem]
        projector.proj_lr = {"sudoku": 1, "cube": 1}[problem]
        projector.rtol = {"sudoku": 2e-2, "cube": 4e-2}[problem]
        projector.auto = True
    else:
        projector = None

    batchSz, testBatchSz, testPct, nEpoch = 40, 40, 0.1, 100
    gpu = (trial_num - 1) % torch.cuda.device_count()
    # gpu = 0

    print('===> Initializing CUDA')

    assert torch.cuda.is_available()
    torch.cuda.set_device(torch.device("cuda:{}".format(gpu)))
    print('===> Using', torch.cuda.get_device_name(gpu))
    print('===> Using', "cuda:{}".format(gpu))
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
        if model == "AutoSymSATNet" and projector.auto and projector.proj_period + 1 == epoch:
            S = sat.S.detach().cpu()
            C = S @ S.T
            projector = symfind.sym_find(C[1:, 1:], rtol = projector.rtol)
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


def main(trial_num, problem_num, model_num):
    problems = ["sudoku", "cube"]
    models = ["SATNet-Low", "SATNet-Full", "SATNet-Low-300aux", "SATNet-Full-300aux", "Proj-Mild", "Proj-Sharp", "SymSATNet", "AutoSymSATNet"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_num', type=int, default=trial_num)
    parser.add_argument('--problem', type=str, default=problems[problem_num])
    parser.add_argument('--model', type=str, default=models[model_num])
    args = parser.parse_args(args = [])
    
    assert args.problem in problems
    assert args.model in models

    trial(args.problem, args.model, args.trial_num)


if __name__=='__main__':
    for problem_num in [0]:
        for trial_num in [3, 6]:
            for model_num in [7, 6, 1, 3]:
                main(trial_num = trial_num, problem_num = problem_num, model_num = model_num)


# if __name__ == "__main__":
#     is_sudoku = True
#     projector = Sudoku() if is_sudoku else Cube()
#     C = torch.randn(projector.dim, projector.dim)
#     C = projector.proj(C)

#     grammar = symfind_test.sym_find(C, False)
#     print(grammar)

# if __name__ == "__main__":
#     m, n = 2, 3
#     Cm = Cyclic(dim = m)
#     Cn = Cyclic(dim = n)
#     Sm = Perm(dim = m)
#     Sn = Perm(dim = n)

#     G = Cube().perm_move
#     D = torch.rand(G.dim, G.dim)
#     D_proj = G.proj(D)

#     draw(D_proj)
#     result = symfind.sym_find(D_proj, 1e-2)
#     print(result)
#     print(result.proj_error(D_proj))
 
    # G = reduce(Sum, [Cm] * n)
    # rtol = 0.02
    # n_perm = 60

    # G = Wreath(Wreath(Perm(dim = 3), Perm(dim = 3)), Perm(dim = 3))
    # rtol = 0.01
    # n_perm = 60

    # G = Wreath(Perm(dim = 3), Perm(dim = 10))
    # rtol = 0.02
    # n_perm = 60

    # G = Sum(Wreath(Perm(dim = 3), Perm(dim = 3)), Cyclic(dim = 3))
    # rtol = 0.04
    # n_perm = 60

    # G = Sum(Cm, Sum(Cm, Cm))
    # D = torch.randn(G.dim, G.dim)
    # D = G.proj(D)
    # draw(D, dpi = 800, save = "original")

    # D = swap(D, 3, 10)
    # D = swap(D, 6, 9)
    # D = swap(D, 5, 8)
    # D = swap(D, 4, 7)
    # D = swap(D, 8, 10)
    # D = swap(D, 11, 5)
    # draw(D, dpi = 800, save = "shuffled")


#     G = Kron(Kron(Perm(dim = 2), Perm(dim = 2)), Perm(dim = 2))
#     rtol = 0.01
#     n_perm = 0

#     num_trials = 1000
#     sub_correct = 0
#     correct = 0

#     for _ in range(num_trials):
#         D = torch.randn(G.dim, G.dim)
#         D_proj = G.proj(D)

#         noise = torch.rand(D.shape)
#         noise_ratio = 5e-3
#         D_noise = D_proj + noise * noise_ratio
#         D_perm, D_proj, D_noise = shuffle_all(torch.stack([D, D_proj, D_noise]), n_perm)

#         result = symfind.sym_find(D_noise, rtol = rtol)
#         D_result = result.proj(D_perm)
#         D_result_proj = result.proj(D_proj)

#         if torch.allclose(D_proj, D_result):
#             correct += 1
        
#         if torch.allclose(D_proj, D_result_proj):
#             sub_correct += 1

# print(correct)
# print(sub_correct)

# if __name__ == "__main__":
#     with open("new_results/cube_trial_1/SATNet-Full/layers/10.pt", 'rb') as f:
#         S1 = torch.load(f).detach().cpu()
#         draw(0.2 ** (S1 @ S1.T))

#     with open("trash/sudoku_trial_1/SATNet-Full/layers/5.pt", 'rb') as f:
#         S2 = torch.load(f).detach().cpu()
#         draw(0.2 ** (S2 @ S2.T))

# if __name__ == "__main__":
    # with open("new_results/sudoku_trial_1/SATNet-Full/layers/10.pt", "rb") as f:
    # with open("new_results/cube_trial_1/SATNet-Full/layers/20.pt", "rb") as f:
    #     S = torch.load(f).detach().cpu()
    #     C = S @ S.T
    #     C = C[1:, 1:]
    #     draw(C)

    # result = symfind.sym_find(C, 4e-2)
    # print(result)
# %%
