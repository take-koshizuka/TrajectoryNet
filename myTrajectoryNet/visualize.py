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

def visualize(cfg, data, zs, output_path, N=500):
    fig, ax = plt.subplots(1,1)
    X = []
    labels = []
    label_set = ["Day 00-03", "Day 06-09", "Day 12-15", "Day 18-21", "Day 24-27"]
    for i, tp in enumerate(cfg['timepoints']):
        X.append(data.get_data()[data.sample_index(N, tp)])
        labels.extend([ label_set[i] ]*N)
    X = np.stack(X, 0).reshape(-1, 5)
    s = 200 / np.sqrt(10000)
    scprep.plot.scatter2d(X, c=labels, figsize=(30,15), cmap="Spectral", s=s,
                        ticks=True, label_prefix="PC", ax=ax, title='Trajectories')

    for i in range(100):
        ax.plot(zs[:,i,0], zs[:,i,1])
    
    plt.savefig(output_path, dpi=300)

def visualize_data_point(cfg, data, zs, idx, output_path, N=500):
    fig, ax = plt.subplots(1,1)
    label_set = ["Day 00-03", "Day 06-09", "Day 12-15", "Day 18-21", "Day 24-27"]
    
    zss = zs[idx, :, :2][cfg['leaveout_timepoint']][:N]
    X = [ 
        data.get_data()[data.get_times() == cfg['timepoints'][cfg['leaveout_timepoint']]][:N, :2], 
        zss
    ]
    X = np.stack(X, 0).reshape(-1, 2)
    assert len(X.shape) == 2
    labels = ([ label_set[cfg['leaveout_timepoint']] + " (ref) " ] * N) + ([ label_set[cfg['leaveout_timepoint']] + " (gen) " ] * N) 
    s = 200 / np.sqrt(10000)
    scprep.plot.scatter2d(X, c=labels, figsize=(30,15), cmap="Spectral", s=s,
                        ticks=True, label_prefix="PC", ax=ax, title='Data distribution')

    plt.savefig(output_path, dpi=300)

def integrate_backwards(cfg, end_samples, model_g, model_f, savedir, ntimes=100, memory=0.1, device='cpu'):
    """ Integrate some samples backwards and save the results.
    """
    integration_times_g =  torch.tensor([0.0, cfg['model_g']['time_scale']])
    with torch.no_grad():
        yT = torch.from_numpy(end_samples).type(torch.float32).to(device)
        xT = model_g(yT, integration_times=integration_times_g)

        ys = [yT]
        xs = [xT]

        int_tps = np.linspace(cfg['int_tps'][0], cfg['int_tps'][-1], ntimes)

        k = -1
        idx = []
        for i, itp in enumerate(int_tps[::-1][:-1]):
            # tp counts down from last
            timescale = int_tps[1] - int_tps[0]
            integration_times_f = torch.tensor([itp, itp - timescale]).type(torch.float32).to(device)

            if k < len(cfg['int_tps']) and cfg['int_tps'][k] >= itp:
                k -= 1
                idx.append(i)
            
            # transform to previous timepoint
            xt = model_f(xs[-1], integration_times=integration_times_f)
            yt = model_g(xt, integration_times=integration_times_g, reverse=True)
    
            xs.append(xt)
            ys.append(yt)
    

        if len(idx) < len(cfg['int_tps']):
            assert len(idx) == (len(cfg['int_tps'])-1)
            idx.append(len(xs) - 1)
        
        xs = torch.stack(xs, 0).cpu().numpy()
        ys = torch.stack(ys, 0).cpu().numpy()

        np.save(os.path.join(savedir, 'backward_trajectories_x.npy'), xs)
        np.save(os.path.join(savedir, 'backward_trajectories_y.npy'), ys)
        np.save(os.path.join(savedir, 'backward_markers_idx.npy'), np.array(idx[::-1]))

    print(idx)
    return xs, ys, idx[::-1]

def integrate_forwards(cfg, data, model_g, model_f, savedir, ntimes=100, memory=0.1, device='cpu'):
    n = 500
    z_samples = data.base_sample()(n, *data.get_shape()).to(device)
    # Forward pass through the model / growth model
    integration_times_g =  torch.tensor([0.0, cfg['model_g']['time_scale']])
    with torch.no_grad():
        integration_times_f = torch.tensor([cfg['int_tps'][0], 0.0]).type(torch.float32).to(device)
        x0 = model_f(z_samples, integration_times=integration_times_f, reverse=True)
        y0 = model_g(x0, integration_times=integration_times_g, reverse=True)

        xs = [x0]
        ys = [y0]

        int_tps = np.linspace(cfg['int_tps'][0], cfg['int_tps'][-1], ntimes)
        k = 0
        idx = []
        for i, itp in enumerate(int_tps[1:]): 
            timescale = int_tps[1] - int_tps[0]
            integration_times_f = torch.tensor([itp, itp - timescale]).type(torch.float32).to(device)

            if k < len(cfg['int_tps']) and cfg['int_tps'][k] <= itp:
                k += 1
                idx.append(i)
    
            xt = model_f(xs[-1], integration_times=integration_times_f, reverse=True)
            yt = model_g(xt, integration_times=integration_times_g, reverse=True)

            xs.append(xt)
            yt.append(yt)

        if len(idx) < len(cfg['int_tps']):
            assert len(idx) == (len(cfg['int_tps'])-1)
            idx.append(len(xs) - 1)

        xs = torch.stack(xs).cpu().numpy()
        ys = torch.stack(ys).cpu().numpy()

        np.save(os.path.join(savedir, 'forward_trajectories_x.npy'), xs)
        np.save(os.path.join(savedir, 'forward_trajectories_y.npy'), ys)
        np.save(os.path.join(savedir, 'forward_markers_idx.npy'), np.array(idx))
    
    print(idx)
    return xs, ys, idx

def main(args):
    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    if args.use_cpu:
        device = torch.device("cpu")

    with open(Path(args.dir) / "config.json", 'r') as f:
        cfg = json.load(f)

    
    data = dataset.SCData.factory(cfg['dataset'])

    timepoints = data.get_unique_times()
    cfg['timepoints'] = timepoints

    # Use maximum timepoint to establish integration_times
    # as some timepoints may be left out for validation etc.
    cfg['int_tps'] = (np.arange(max(timepoints) + 1) + 1.0) * cfg['model_f']['time_scale']
    
    regularization_fns_g, regularization_coeffs_g = create_regularization_fns(cfg['model_g']['reg'])
    regularization_fns_f, regularization_coeffs_f = create_regularization_fns(cfg['model_f']['reg'])
    
    model_g = build_model_tabular(cfg['model_g'], data.get_shape()[0], regularization_fns_g).to(device)
    model_f = build_model_tabular(cfg['model_f'], data.get_shape()[0], regularization_fns_f).to(device)

    if cfg['use_growth']:
        growth_model_path = data.get_growth_net_path()
        #growth_model_path = "/home/atong/TrajectoryNet/data/externel/growth_model_v2.ckpt"
        growth_model = torch.load(growth_model_path, map_location=device)

    if cfg['model_g']['reg']['spectral_norm']:
        add_spectral_norm(model_g)
    
    if cfg['model_f']['reg']['spectral_norm']:
        add_spectral_norm(model_f)

    set_cnf_options(cfg['model_g'], model_g)
    set_cnf_options(cfg['model_f'], model_f)


    state_dict = torch.load(args.dir + "/checkpt.pth", map_location=device)
    model_g.load_state_dict(state_dict["state_dict_g"])
    model_f.load_state_dict(state_dict["state_dict_f"])


    if not os.path.exists(os.path.join(args.dir, 'backward_trajectories_x.npy')):
        end_time_data = data.data_dict["pcs"]
        end_time_data = data.get_data()[data.get_times()==np.max(data.get_times())]
        np.random.permutation(end_time_data)
        rand_idx = np.random.randint(end_time_data.shape[0], size=5000)
        end_time_data = end_time_data[rand_idx,:]
        integrate_backwards(cfg, end_time_data, model_g, model_f, args.dir, ntimes=100, device=device)

    if not os.path.exists(os.path.join(args.dir, 'forward_trajectories_x.npy')):
        integrate_forwards(cfg, data, model_g, model_f, args.dir, ntimes=100, device=device)

    print("load trajectories")
    xs = np.load(os.path.join(args.dir, 'backward_trajectories_x.npy'))
    ys = np.load(os.path.join(args.dir, 'backward_trajectories_y.npy'))
    idx = np.load(os.path.join(args.dir, 'backward_markers_idx.npy'))
    
    output_path = os.path.join(args.dir, 'backward_x.png')
    visualize(cfg, xs, output_path)

    output_path = os.path.join(args.dir, 'backward_y.png')
    visualize(cfg, ys, output_path)

    output_path = os.path.join(args.dir, 'backward_data_point.png')
    visualize_data_point(cfg, ys, idx, output_path)

    xs = np.load(os.path.join(args.dir, 'forward_trajectories_x.npy'))
    ys = np.load(os.path.join(args.dir, 'forward_trajectories_y.npy'))
    idx = np.load(os.path.join(args.dir, 'forward_markers_idx.npy'))
    
    output_path = os.path.join(args.dir, 'forward_x.png')
    visualize(cfg, xs, output_path)

    output_path = os.path.join(args.dir, 'forward_y.png')
    visualize(cfg, ys, output_path)

    output_path = os.path.join(args.dir, 'forward_data_point.png')
    visualize_data_point(cfg, ys, idx, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Continuous Normalizing Flow")
    parser.add_argument('-dir', '-d', help="the configuration file for training.", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--no_display_loss", action="store_false")
    args = parser.parse_args()
    
    main(args)
