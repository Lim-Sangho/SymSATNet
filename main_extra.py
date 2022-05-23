# %%
from functools import reduce
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
from utils import symfind
import satnet
import symsatnet


def run(epoch, model, projector, optimizer, loader, logger, figlogger, save_dir, to_train = False):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)
    start.record()
    
    loss_total, err_total = 0, 0
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

    if logger is not None:
        logger.log((epoch, loss_total, err_total, start.elapsed_time(end)))
    if figlogger is not None:
        figlogger.log((epoch, loss_total, err_total))

    
    #print('memory: {:.2f} MB, cached: {:.2f} MB'.format(torch.cuda.memory_allocated()/2.**20, torch.cuda.memory_cached()/2.**20))
    torch.cuda.empty_cache()

    return err_total


def train(epoch, model, projector, optimizer, loader, logger, figlogger, save_dir):
    return run(epoch, model, projector, optimizer, loader, logger, figlogger, save_dir, True)


@torch.no_grad()
def test(epoch, model, projector, optimizer, loader, logger, figlogger, save_dir):
    return run(epoch, model, projector, optimizer, loader, logger, figlogger, save_dir, False)


def trial(problem, model, trial_num, corrupt_num = 0):
    print("===> Trial: {}".format(trial_num))
    print("===> Problem: {}".format(problem))
    print("===> Model: {}".format(model))
    print("===> Setting hyperparameters")

    n = {"sudoku": 729, "cube": 324}[problem]
    data_dir = {"sudoku": "dataset/sudoku_10000", "cube": "dataset/cube_10000"}[problem]
    save_dir = problem + f"_trial_{trial_num}_corrupt_{corrupt_num}/" + model

    rank = {"SATNet-Low": 5/6, "SATNet-Full": 1, "SATNet-Low-300aux": 5/6, "SATNet-Full-300aux": 1,
            "Proj-Mild": 1, "Proj-Sharp": 1, "SymSATNet": 1, "SymSATNet-Auto": 1, "SymSATNet-Val": 1}[model]
    aux = {"SATNet-Low": 0, "SATNet-Full": 0, "SATNet-Low-300aux": 300, "SATNet-Full-300aux": 300,
           "Proj-Mild": 0, "Proj-Sharp": 0, "SymSATNet": 0, "SymSATNet-Auto": 0, "SymSATNet-Val": 0}[model]
    lr = {"SATNet-Low": 2e-3, "SATNet-Full": 2e-3, "SATNet-Low-300aux": 2e-3, "SATNet-Full-300aux": 2e-3,
          "Proj-Mild": 2e-3, "Proj-Sharp": 2e-3, "SymSATNet": 4e-2, "SymSATNet-Auto": 2e-3, "SymSATNet-Val": 2e-3}[model]

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
        projector.rtol = {"sudoku": 2e-2, "cube": 4e-2}[problem]
        projector.auto = True
    elif model == "SymSATNet-Val":
        projector = Id(dim = n)
        projector.proj_period = {"sudoku": 10, "cube": 20}[problem]
        projector.proj_lr = {"sudoku": 1, "cube": 1}[problem]
        projector.rtol = {"sudoku": 2e-2, "cube": 4e-2}[problem]
        projector.auto = True
    else:
        projector = None

    nEpoch = 100
    # gpu = 0
    gpu = (trial_num - 1) % torch.cuda.device_count()

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
    batchSz, testBatchSz, validBatchSz = 40, 40, 40
    testPct, validPct = (0.1, 0.1) if model == "SymSATNet-Val" else (0.1, 0.0)
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
        # i1, i2, j1, j2 = torch.randint(0, 9, (4,))
        # while (i1, j1) == (i2, j2):
        #     i2, j2 = torch.randint(0, 9, (2,))

        for pair in indices:
            entry = torch.argmax(label[pair])
            corrupt = torch.randint(0, r, (1,))
            while corrupt == entry:
                corrupt = torch.randint(0, r, (1,))
            label[pair] = torch.nn.functional.one_hot(corrupt, r)
            if not torch.all(feature[pair] == 0):
                feature[pair] = torch.nn.functional.one_hot(corrupt, r)
        # k1 = torch.argmax(label[i1, j1])
        # k2 = torch.argmax(label[i2, j2])
        # k1_corrupt, k2_corrupt = torch.randint(0, 9, (2,))
        # while k1_corrupt == k1 or k2_corrupt == k2:
        #     k1_corrupt, k2_corrupt = torch.randint(0, 9, (2,))
        # label[i1, j1] = torch.nn.functional.one_hot(k1_corrupt, 9)
        # label[i2, j2] = torch.nn.functional.one_hot(k2_corrupt, 9)
        # if not torch.all(feature[i1, j1] == 0):
        #     feature[i1, j1] = torch.nn.functional.one_hot(k1_corrupt, 9)
        # if not torch.all(feature[i2, j2] == 0):
        #     feature[i2, j2] = torch.nn.functional.one_hot(k2_corrupt, 9)


    is_input = X.sum(dim = 3, keepdim = True).expand_as(X).int().sign()
    X, Y, is_input = X.cuda(), Y.cuda(), is_input.cuda()

    train_set = TensorDataset(X[:nTrain], is_input[:nTrain], Y[:nTrain])
    test_set = TensorDataset(X[nTrain:nTrain+nTest], is_input[nTrain:nTrain+nTest], Y[nTrain:nTrain+nTest])
    if model == "SymSATNet-Val":
        valid_set = TensorDataset(X[nTrain+nTest:], is_input[nTrain+nTest:], Y[nTrain+nTest:])

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
    fig, axes = plt.subplots(1,3, figsize=(10,4))
    plt.subplots_adjust(wspace=0.4)
    figtrain_logger = FigLogger(fig, axes[0], 'Traininig')
    figtest_logger = FigLogger(fig, axes[1], 'Testing')

    for epoch in range(1, nEpoch+1):
        if (model in ["SymSATNet-Auto", "SymSATNet-Val"]) and projector.auto and projector.proj_period + 1 == epoch:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing = True)
            end = torch.cuda.Event(enable_timing = True)
            start.record()

            S = sat.S.detach().cpu()
            C = S @ S.T
            projector = symfind.sym_find(C[1:, 1:], rtol = projector.rtol)

            # Find proper subgroup with validation error
            if model == "SymSATNet-Val":
                valid_loader = tqdm(DataLoader(valid_set, batch_size = validBatchSz))
                valid_args = [epoch, sat, projector, None, valid_loader, None, None, save_dir]
                valid_err = test(*valid_args)
                projector = validation(projector, S, lambda G: G, valid_err, valid_args)

            # print(projector)
            basis = projector.basis()
            upper = proj_orbit(C[0:1, 1:], Id(dim = 1), projector)[0]
            coeff = coordinates(C[1:, 1:], basis)

            sat = symsatnet.SymSATNet(n, basis).cuda()
            sat.coeff = torch.nn.Parameter(coeff.cuda())
            sat.upper = torch.nn.Parameter(upper.cuda())

            end.record()
            torch.cuda.synchronize()

            print('S is Projected: {}'.format(projector))
            print("Elapsed time: {}".format(start.elapsed_time(end)))
            with open(save_dir + "/logs/proj.csv", "a") as f:
                f.write(f"{epoch},{projector},{start.elapsed_time(end)}\n")
                f.close()

            projector.proj_period = float("inf")
            projector.proj_lr = 0
            projector.auto = True

        train_loader = tqdm(DataLoader(train_set, batch_size = batchSz))
        train(epoch, sat, projector, optimizer, train_loader, train_logger, figtrain_logger, save_dir)
        test_loader = tqdm(DataLoader(test_set, batch_size = testBatchSz))
        test(epoch, sat, projector, optimizer, test_loader, test_logger, figtest_logger, save_dir)
        display(fig)


def validation(projector, S, frame, valid_err, valid_args):
    # print(projector)

    new_S = frame(projector).proj_S(S, 1.0)
    new_S.requires_grad = True
    valid_args[1].S = torch.nn.Parameter(new_S.cuda())
    err = test(*valid_args)
    if err <= valid_err - 0.1: # 0.1 if corrupt_num == 1 else 0.2
        return projector

    if isinstance(projector, Id) or isinstance(projector, Cyclic) or isinstance(projector, Perm):
        return Id(dim = projector.dim)

    if isinstance(projector, Kron):
        G1 = projector.G1
        G2 = projector.G2
        frame1 = lambda G: frame(Kron(G, Id(dim = G2.dim)).permute(projector.perm))
        frame2 = lambda G: frame(Kron(Id(dim = G1.dim), G).permute(projector.perm))
        new_G1 = validation(G1, S, frame1, valid_err, valid_args)
        new_G2 = validation(G2, S, frame2, valid_err, valid_args)

        return Kron(new_G1, new_G2).permute(projector.perm)

    if isinstance(projector, Sum):
        G1 = projector.G1
        G2 = projector.G2
        frame1 = lambda G: frame(Sum(G, Id(dim = G2.dim)).permute(projector.perm))
        frame2 = lambda G: frame(Sum(Id(dim = G1.dim), G).permute(projector.perm))
        new_G1 = validation(G1, S, frame1, valid_err, valid_args)
        new_G2 = validation(G2, S, frame2, valid_err, valid_args)

        return Sum(new_G1, new_G2).permute(projector.perm)

    if isinstance(projector, Wreath):
        G1 = projector.G1
        G2 = projector.G2

        new_projector = Kron(G2, G1)
        new_S = frame(new_projector).proj_S(S, 1.0)
        new_S.requires_grad = True
        valid_args[1].S = torch.nn.Parameter(new_S.cuda())
        err = test(*valid_args)
        if err <= valid_err:
            return projector

        frame1 = lambda G: frame(Wreath(G, Id(dim = G2.dim)).permute(projector.perm))
        frame2 = lambda G: frame(Wreath(Id(dim = G1.dim), G).permute(projector.perm))
        new_G1 = validation(G1, S, frame1, valid_err, valid_args)
        new_G2 = validation(G2, S, frame2, valid_err, valid_args)

        return Wreath(new_G1, new_G2).permute(projector.perm)


def main(problem_num, model_num, trial_num, corrupt_num):
    problems = ["sudoku", "cube"]
    models = ["SATNet-Low", "SATNet-Full", "SATNet-Low-300aux", "SATNet-Full-300aux", "Proj-Mild", "Proj-Sharp", "SymSATNet", "SymSATNet-Auto", "SymSATNet-Val"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default=problems[problem_num])
    parser.add_argument('--model', type=str, default=models[model_num])
    parser.add_argument('--trial_num', type=int, default=trial_num)
    parser.add_argument('--corrupt_num', type=int, default=corrupt_num)
    args = parser.parse_args(args = [])
    
    assert args.problem in problems
    assert args.model in models

    trial(args.problem, args.model, args.trial_num, args.corrupt_num)

if __name__ == '__main__':
    for problem_num in [0]:
        for trial_num in [3]:
            for model_num in [8]:
                for corrupt_num in [3]:
                    main(problem_num, model_num, trial_num, corrupt_num)

# if __name__ == "__main__":
#     with open("cube_trial_2_corrupt_1/SATNet-Full/layers/20.pt", "rb") as f:
#         S = torch.load(f).detach().cpu()
#         C = S @ S.T
#         C = C[1:, 1:]

#     draw(C)
#     print(symfind.sym_find(C, 4e-2))

# %%