import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import argparse
import json
import scprep
from pathlib import Path

from lib.growth_net import GrowthNet
import lib.utils as utils
from lib.viz_scrna import trajectory_to_video, save_vectors
from lib.viz_scrna import (
    save_trajectory_density,
    save_2d_trajectory,
    save_2d_trajectory_v2,
)

# from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters
from train_misc import count_total_time
from train_misc import add_spectral_norm, spectral_norm_power_iteration
from train_misc import create_regularization_fns, get_regularization
from train_misc import append_regularization_to_log
from train_misc import build_model_tabular
import random
import eval_utils
import dataset
from main import get_transforms
from lib.visualize_flow import visualize_transform
import phate
import sklearn.preprocessing

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def visualize(args, zs, output_path, N=500):
    fig, ax = plt.subplots(1,1)
    X = []
    labels = []
    label_set = ["Day 00-03", "Day 06-09", "Day 12-15", "Day 18-21", "Day 24-27"]
    for i, tp in enumerate(args.timepoints):
        X.append(args.data.get_data()[args.data.sample_index(N, tp)])
        labels.extend([ label_set[i] ]*N)
    X = np.stack(X, 0).reshape(-1, 5)
    s = 200 / np.sqrt(10000)
    scprep.plot.scatter2d(X, c=labels, figsize=(30,15), cmap="Spectral", s=s,
                        ticks=True, label_prefix="PC", ax=ax, title='Trajectories')

    for i in range(100):
        ax.plot(zs[:,i,0], zs[:,i,1])
    
    plt.savefig(output_path, dpi=300)

def visualize_data_point(args, zs, idx, output_path, N=500):
    fig, ax = plt.subplots(1,1)
    label_set = ["Day 00-03", "Day 06-09", "Day 12-15", "Day 18-21", "Day 24-27"]
    
    zss = zs[idx, :, :2][args.leaveout_timepoint][:N]
    X = [ 
        args.data.get_data()[args.data.get_times() == args.timepoints[args.leaveout_timepoint]][:N, :2], 
        zss
    ]
    X = np.stack(X, 0).reshape(-1, 2)
    assert len(X.shape) == 2
    labels = ([ label_set[args.leaveout_timepoint] + " (ref) " ] * N) + ([ label_set[args.leaveout_timepoint] + " (gen) " ] * N) 
    s = 200 / np.sqrt(10000)
    scprep.plot.scatter2d(X, c=labels, figsize=(30,15), cmap="Spectral", s=s,
                        ticks=True, label_prefix="PC", ax=ax, title='Data distribution')

    plt.savefig(output_path, dpi=300)

def integrate_backwards(args, end_samples, model, savedir, ntimes=100, memory=0.1, device='cpu'):
    """ Integrate some samples backwards and save the results.
    """
    with torch.no_grad():
        z = torch.from_numpy(end_samples).type(torch.float32).to(device)
        zero = torch.zeros(z.shape[0], 1).to(z)
        cnf = model.chain[0]

        zs = [z]
        deltas = []
        int_tps = np.linspace(args.int_tps[0], args.int_tps[-1], ntimes)

        k = -1
        idx = []
        for i, itp in enumerate(int_tps[::-1][:-1]):
            # tp counts down from last
            timescale = int_tps[1] - int_tps[0]
            integration_times = torch.tensor([itp, itp - timescale])
            # integration_times = torch.tensor([np.linspace(itp - args.time_scale, itp, ntimes)])
            integration_times = integration_times.type(torch.float32).to(device)

            if k < len(args.int_tps) and args.int_tps[k] >= itp:
                k -= 1
                idx.append(i)
            
            # transform to previous timepoint
            z, delta_logp = cnf(zs[-1], zero, integration_times=integration_times)
            zs.append(z)
            deltas.append(delta_logp)

        if len(idx) < len(args.int_tps):
            assert len(idx) == (len(args.int_tps)-1)
            idx.append(len(zs) - 1)
        
        zs = torch.stack(zs, 0)
        zs = zs.cpu().numpy()
        np.save(os.path.join(savedir, 'backward_trajectories.npy'), zs)
        np.save(os.path.join(savedir, 'backward_markers_idx.npy'), np.array(idx[::-1]))

    print(idx)
    return zs, idx[::-1]

def integrate_forwards(args, data, model, savedir, ntimes=100, memory=0.1, device='cpu'):
    n = 500
    z_samples = data.base_sample()(n, *data.get_shape()).to(device)
    # Forward pass through the model / growth model
    with torch.no_grad():
        integration_times = torch.tensor([args.int_tps[0], 0.0]).type(torch.float32).to(device)
        print(integration_times)
        x = model(z_samples, integration_times=integration_times, reverse=True)
        
        zs = [x]
        int_tps = np.linspace(args.int_tps[0], args.int_tps[-1], ntimes)
        k = 0
        idx = []
        for i, itp in enumerate(int_tps[1:]): 
            timescale = int_tps[1] - int_tps[0]
            integration_times = torch.tensor([itp, itp - timescale])
            integration_times = integration_times.type(torch.float32).to(device)

            if k < len(args.int_tps) and args.int_tps[k] <= itp:
                k += 1
                idx.append(i)
    
            z = model(zs[-1], integration_times=integration_times, reverse=True)
            zs.append(z)

        if len(idx) < len(args.int_tps):
            assert len(idx) == (len(args.int_tps)-1)
            idx.append(len(zs) - 1)

        zs = torch.stack(zs)
        zs = zs.cpu().numpy()

        np.save(os.path.join(savedir, 'forward_trajectories.npy'), zs)
        np.save(os.path.join(savedir, 'forward_markers_idx.npy'), np.array(idx))
    
    print(idx)
    return zs, idx

def main(args):
    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    if args.use_cpu:
        device = torch.device("cpu")
    
    args.data = dataset.SCData.factory(args.dataset, args)
    data = args.data

    args.timepoints = data.get_unique_times()

    # Use maximum timepoint to establish integration_times
    # as some timepoints may be left out for validation etc.
    args.int_tps = (np.arange(max(args.timepoints) + 1) + 1.0) * args.time_scale

    print(args.int_tps)

    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = build_model_tabular(args, data.get_shape()[0], regularization_fns).to(
        device
    )
    if args.use_growth:
        growth_model_path = data.get_growth_net_path()
        #growth_model_path = "/home/atong/TrajectoryNet/data/externel/growth_model_v2.ckpt"
        growth_model = torch.load(growth_model_path, map_location=device)
    if args.spectral_norm:
        add_spectral_norm(model)
    set_cnf_options(args, model)

    state_dict = torch.load(args.save + "/checkpt.pth", map_location=device)
    model.load_state_dict(state_dict["state_dict"])


    if not os.path.exists(os.path.join(args.save, 'backward_trajectories.npy')):
        end_time_data = data.data_dict["pcs"]
        end_time_data = data.get_data()[data.get_times()==np.max(data.get_times())]
        np.random.permutation(end_time_data)
        rand_idx = np.random.randint(end_time_data.shape[0], size=5000)
        end_time_data = end_time_data[rand_idx,:]
        integrate_backwards(args, end_time_data, model, args.save, ntimes=100, device=device)

    if not os.path.exists(os.path.join(args.save, 'forward_trajectories.npy')):
        integrate_forwards(args, data, model, args.save, ntimes=100, device=device)

    print("load trajectories")
    zs = np.load(os.path.join(args.save, 'backward_trajectories.npy'))
    idx = np.load(os.path.join(args.save, 'backward_markers_idx.npy'))
    
    output_path = os.path.join(args.save, 'backward.png')
    visualize(args, zs, output_path)

    output_path = os.path.join(args.save, 'backward_data_point.png')
    visualize_data_point(args, zs, idx, output_path)

    zs = np.load(os.path.join(args.save, 'forward_trajectories.npy'))
    idx = np.load(os.path.join(args.save, 'forward_markers_idx.npy'))

    output_path = os.path.join(args.save, 'forward.png')
    visualize(args, zs, output_path)

    output_path = os.path.join(args.save, 'forward_data_point.png')
    visualize_data_point(args, zs, idx, output_path)


if __name__ == "__main__":
    from parse import parser
    fix_seed(10)
    args = parser.parse_args()
    main(args)
