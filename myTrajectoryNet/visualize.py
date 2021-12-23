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

import eval_utils
import dataset
from main import get_transforms
from lib.visualize_flow import visualize_transform
import phate
import sklearn.preprocessing

import os

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def load_data():
    download_path = "../data"
    if not os.path.isdir(os.path.join(download_path, "scRNAseq", "T0_1A")):
        # need to download the data
        scprep.io.download.download_and_extract_zip(
            "https://md-datasets-public-files-prod.s3.eu-west-1.amazonaws.com/"
            "5739738f-d4dd-49f7-b8d1-5841abdbeb1e",
            download_path)
    sparse=True
    T1 = scprep.io.load_10X(os.path.join(download_path, "scRNAseq", "T0_1A"), sparse=sparse, gene_labels='both')
    T2 = scprep.io.load_10X(os.path.join(download_path, "scRNAseq", "T2_3B"), sparse=sparse, gene_labels='both')
    T3 = scprep.io.load_10X(os.path.join(download_path, "scRNAseq", "T4_5C"), sparse=sparse, gene_labels='both')
    T4 = scprep.io.load_10X(os.path.join(download_path, "scRNAseq", "T6_7D"), sparse=sparse, gene_labels='both')
    T5 = scprep.io.load_10X(os.path.join(download_path, "scRNAseq", "T8_9E"), sparse=sparse, gene_labels='both')
    filtered_batches = []
    for batch in [T1, T2, T3, T4, T5]:
        batch = scprep.filter.filter_library_size(batch, percentile=20, keep_cells='above')
        batch = scprep.filter.filter_library_size(batch, percentile=75, keep_cells='below')
        filtered_batches.append(batch)
    del T1, T2, T3, T4, T5 
    EBT_counts, sample_labels = scprep.utils.combine_batches(
        filtered_batches, 
        ["Day 00-03", "Day 06-09", "Day 12-15", "Day 18-21", "Day 24-27"],
        append_to_cell_names=True
    )
    phate_operator = phate.PHATE(n_jobs=-2, random_state=42)
    
    EBT_counts = scprep.filter.filter_rare_genes(EBT_counts, min_cells=10)
    EBT_counts = scprep.normalize.library_size_normalize(EBT_counts)
    mito_genes = scprep.select.get_gene_set(EBT_counts, starts_with="MT-")
    EBT_counts, sample_labels = scprep.filter.filter_gene_set_expression(
        EBT_counts, sample_labels, genes=mito_genes, percentile=90, keep_cells='below')
    EBT_counts = scprep.transform.sqrt(EBT_counts)
    Y_phate = phate_operator.fit_transform(EBT_counts)
    return phate_operator, sample_labels

def visualize(phate_operator, sample_labels, zs, output_path):
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(phate_operator.graph.data_nu)
    zss = zs * scaler.scale_[:5] + scaler.mean_[:5]

    fig, ax = plt.subplots(1,1)
    scprep.plot.scatter2d(phate_operator.graph.data_nu, c=sample_labels, figsize=(30,15), cmap="Spectral",
                        ticks=True, label_prefix="PC", ax=ax, title='Trajectories for 200 cells from last timepoint')

    for i in range(100):
        ax.plot(zss[:,i,0], zss[:,i,1])

    plt.savefig(output_path, dpi=300)

def visualize_generated_sample(phate_operator, sample_labels, zs, idx, output_path):
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(phate_operator.graph.data_nu)
    zss = zs * scaler.scale_[:5] + scaler.mean_[:5]
    fig, ax = plt.subplots(1,1)
    
    T = len(idx) - 1
    data_n = zss.shape[1]
    X = zss[idx, :, :2].reshape(-1, 2)
    c = (["Day 24-27"] * data_n) + (["Day 18-21"] * data_n) + (["Day 12-15"] * data_n) + (["Day 06-09"] * data_n) + (["Day 00-03"] * data_n)
    scprep.plot.scatter2d(X, c=c, figsize=(30,15), cmap="Spectral",
                        ticks=True, label_prefix="PC", ax=ax, title='Trajectories for 200 cells from last timepoint')
    for i in range(100):
        ax.plot(zss[:,i,0], zss[:,i,1])
        
    plt.savefig(output_path, dpi=300)

def visualize_data_point(phate_operator, sample_labels, zs, idx, output_path):
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(phate_operator.graph.data_nu)
    zss = zs * scaler.scale_[:5] + scaler.mean_[:5]
    fig, ax = plt.subplots(1,1)
    N = 500

    s = 200 / np.sqrt(len(phate_operator.graph.data_nu))
    label_set = ["Day 00-03", "Day 06-09", "Day 12-15", "Day 18-21", "Day 24-27"]
    X = []
    labels = []
    for label in label_set:
        X.append(phate_operator.graph.data_nu[sample_labels == label][:N])
        labels.extend([label]*N)
    X = np.stack(X, 0)
    X = X.reshape(-1, X.shape[2])
    
    scprep.plot.scatter2d(X, c=labels, figsize=(30,15), cmap="magma", s=s,
                        ticks=True, label_prefix="PC", ax=ax, title='Trajectories for 200 cells from last timepoint')
    
    X = zss[idx, :, :2]
    ax.scatter(X[0, :N, 0], X[0, :N, 1], s=s, alpha=0.8, cmap="inferno")
    ax.scatter(X[1, :N, 0], X[1, :N, 1], s=s, alpha=0.8, cmap="inferno")
    ax.scatter(X[2, :N, 0], X[2, :N, 1], s=s, alpha=0.8, cmap="inferno")
    ax.scatter(X[3, :N, 0], X[3, :N, 1], s=s, alpha=0.8, cmap="inferno")
    ax.scatter(X[4, :N, 0], X[4, :N, 1], s=s, alpha=0.8, cmap="inferno")

    plt.savefig(output_path, dpi=300)

def integrate_backwards(cfg, end_samples, model_g, model_f, savedir, ntimes=100, memory=0.1, device='cpu'):
    """ Integrate some samples backwards and save the results.
    """
    with torch.no_grad():
        z = torch.from_numpy(end_samples).type(torch.float32).to(device)
        zero = torch.zeros(z.shape[0], 1).to(z)
        cnf = model_f.chain[0]

        x = model_g(z, reverse=False)
        xs = [x]
        ys = [z]
        deltas = []
        
        int_tps = np.linspace(cfg['int_tps'][0], cfg['int_tps'][-1], ntimes)
        k = -1
        idx = []
        for i, itp in enumerate(int_tps[::-1][:-1]):
            # tp counts down from last
            timescale = int_tps[1] - int_tps[0]
            integration_times = torch.tensor([itp - timescale, itp])
            # integration_times = torch.tensor([np.linspace(itp - args.time_scale, itp, ntimes)])
            integration_times = integration_times.type(torch.float32).to(device)

            if k < len(cfg['int_tps']) and cfg['int_tps'][k] >= itp:
                k -= 1
                idx.append(i)
                
            # transform to previous timepoint
            next_x, delta_logp = cnf(xs[-1], zero, integration_times=integration_times)
            y = model_g(next_x, reverse=True)
            
            xs.append(next_x)
            ys.append(y)
            deltas.append(delta_logp)

        if len(idx) < len(cfg['int_tps']):
            idx.append(len(xs) - 1)

        ys = torch.stack(ys, 0)
        ys = ys.cpu().numpy()
        xs = torch.stack(xs, 0)
        xs = xs.cpu().numpy()

        np.save(os.path.join(savedir, 'backward_trajectories_x.npy'), xs)
        np.save(os.path.join(savedir, 'backward_trajectories_y.npy'), ys)
        np.save(os.path.join(savedir, 'backward_markers_idx.npy'), np.array(idx))
        return xs, ys, idx

def integrate_forwards(cfg, data, model_g, model_f, savedir, ntimes=100, memory=0.1, device='cpu'):
    n = 500
    z_samples = data.base_sample()(n, *data.get_shape()).to(device)
    # Forward pass through the model / growth model
    with torch.no_grad():
        integration_times = torch.tensor([0.0, cfg['int_tps'][0]]).type(torch.float32).to(device)
        x_1 = model_f(z_samples, integration_times=integration_times, reverse=True)
        y_1 = model_g(x_1, reverse=True)
        xs = [x_1]
        ys = [y_1]
        int_tps = np.linspace(cfg['int_tps'][0], cfg['int_tps'][-1], ntimes)
        k = 0
        idx = []
        for i, itp in enumerate(int_tps[1:]): 
            timescale = int_tps[1] - int_tps[0]
            integration_times = torch.tensor([itp - timescale, itp])
            integration_times = integration_times.type(torch.float32).to(device)

            if k < len(cfg['int_tps']) and cfg['int_tps'][k] <= itp:
                k += 1
                idx.append(i)
    
            next_x = model_f(xs[-1], integration_times=integration_times, reverse=True)
            y = model_g(next_x, reverse=True)
            ys.append(y)
            xs.append(next_x)

        if len(idx) < len(cfg['int_tps']):
            idx.append(len(xs) - 1)

        xs = torch.stack(xs)
        xs = xs.cpu().numpy()

        ys = torch.stack(ys)
        ys = ys.cpu().numpy()
        
        np.save(os.path.join(savedir, 'forward_trajectories_x.npy'), xs)
        np.save(os.path.join(savedir, 'forward_trajectories_y.npy'), ys)
        np.save(os.path.join(savedir, 'forward_markers_idx.npy'), np.array(idx))
    
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

    phate_operator, sample_labels = load_data()
    print("load trajectories")
    xs = np.load(os.path.join(args.dir, 'backward_trajectories_x.npy'))
    ys = np.load(os.path.join(args.dir, 'backward_trajectories_y.npy'))
    idx = np.load(os.path.join(args.dir, 'backward_markers_idx.npy'))
    
    output_path = os.path.join(args.dir, 'backward_x.png')
    visualize(phate_operator, sample_labels, xs, output_path)

    output_path = os.path.join(args.dir, 'backward_xg.png')
    visualize_generated_sample(phate_operator, sample_labels, xs, idx, output_path)

    output_path = os.path.join(args.dir, 'backward_data_point_x.png')
    visualize_data_point(phate_operator, sample_labels, xs, idx, output_path)

    output_path = os.path.join(args.dir, 'backward_y.png')
    visualize(phate_operator, sample_labels, ys, output_path)

    output_path = os.path.join(args.dir, 'backward_yg.png')
    visualize_generated_sample(phate_operator, sample_labels, ys, idx, output_path)

    output_path = os.path.join(args.dir, 'backward_data_point_y.png')
    visualize_data_point(phate_operator, sample_labels, ys, idx, output_path)

    xs = np.load(os.path.join(args.dir, 'forward_trajectories_x.npy'))
    ys = np.load(os.path.join(args.dir, 'forward_trajectories_y.npy'))
    idx = np.load(os.path.join(args.dir, 'forward_markers_idx.npy'))

    output_path = os.path.join(args.dir, 'forward_x.png')
    visualize(phate_operator, sample_labels, xs, output_path)

    output_path = os.path.join(args.dir, 'forward_xg.png')
    visualize_generated_sample(phate_operator, sample_labels, xs, idx, output_path)

    output_path = os.path.join(args.dir, 'forward_data_point_x.png')
    visualize_data_point(phate_operator, sample_labels, xs, idx, output_path)

    output_path = os.path.join(args.dir, 'forward_y.png')
    visualize(phate_operator, sample_labels, ys, output_path)

    output_path = os.path.join(args.dir, 'forward_yg.png')
    visualize_generated_sample(phate_operator, sample_labels, ys, idx, output_path)

    output_path = os.path.join(args.dir, 'forward_data_point_y.png')
    visualize_data_point(phate_operator, sample_labels, ys, idx, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Continuous Normalizing Flow")
    parser.add_argument('-dir', '-d', help="the configuration file for training.", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--no_display_loss", action="store_false")
    args = parser.parse_args()
    
    main(args)
