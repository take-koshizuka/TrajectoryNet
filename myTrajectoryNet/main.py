""" main.py

Learns ODE from scrna data

"""
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import random
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from lib.growth_net import GrowthNet
import lib.utils as utils
from lib.visualize_flow import visualize_transform
from lib.viz_scrna import save_trajectory, trajectory_to_video, save_vectors
from lib.viz_scrna import save_trajectory_density


# from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters
from train_misc import count_total_time
from train_misc import add_spectral_norm, spectral_norm_power_iteration
from train_misc import create_regularization_fns, get_regularization
from train_misc import append_regularization_to_log
from train_misc import build_model_tabular

import dataset

matplotlib.use("Agg")


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_transforms(cfg, model_g, model_f, integration_times, device):
    """
    Given a list of integration points,
    returns a function giving integration times
    """

    # tは順方向
    def sample_fn(z, logpz=None):
        int_list = [
            torch.tensor([it - cfg['time_scale'], it]).type(torch.float32).to(device)
            for it in integration_times
        ]
        if logpz is not None:
            # TODO this works right?
            for it in int_list:
                z, logpz = model_f(z, logpz, integration_times=it, reverse=True)
            y, logpy = model_g(z, logpz, reverse=True)
            return y, logpy
        else:
            for it in int_list:
                z = model_f(z, integration_times=it, reverse=True)
            y = model_g(z, reverse=True)
            return y
    # tは逆方向
    def density_fn(y, logpy=None):
        int_list = [
            torch.tensor([it - cfg['time_scale'], it]).type(torch.float32).to(device)
            for it in integration_times[::-1]
        ]
        if logpy is not None:
            x, logpx = model_g(y, logpy, reverse=False)
            for it in int_list:
                x, logpx = model_f(x, logpx, integration_times=it, reverse=False)
            y, logpy = model_g(x, logpx, reverse=True)
            return y, logpy
        else:
            x = model_g(y, reverse=False)
            for it in int_list:
                x = model_f(x, integration_times=it, reverse=False)
            y = model_g(x, reverse=True)
            return y

    return sample_fn, density_fn


def compute_loss(cfg, data, model_g, model_f, growth_model, logger, device):
    """
    Compute loss by integrating backwards from the last time step
    At each time step integrate back one time step, and concatenate that
    to samples of the empirical distribution at that previous timestep
    repeating over and over to calculate the likelihood of samples in
    later timepoints iteratively, making sure that the ODE is evaluated
    at every time step to calculate those later points.

    The growth model is a single model of time independent cell growth /
    death rate defined as a variation from uniform.
    """

    # Backward pass accumulating losses, previous state and deltas
    deltas_pypx = []
    deltas_px = []
    zs = []
    z = None
    interp_loss = 0.0

    integration_times_g =  torch.tensor([0.0, cfg['model_g']['time_scale']])
    for i, (itp, tp) in enumerate(zip(cfg['int_tps'][::-1], cfg['timepoints'][::-1])):
        # tp counts down from last

        # load data and add noise
        idx = data.sample_index(cfg['batch_size'], tp)
        # x[t] -> x[t-1] -> ... 
        y = data.get_data()[idx]
        if cfg['training_noise'] > 0.0:
            y += np.random.randn(*y.shape) * cfg['training_noise']
        y = torch.from_numpy(y).type(torch.float32).to(device)

        zero = torch.zeros(y.shape[0], 1).to(y)
        
        x, delta_logpypx = model_g(y, zero, integration_times=integration_times_g)
        deltas_pypx.append(delta_logpypx)

        if i > 0:
            x = torch.cat((z, x))
            zs.append(z)

        integration_times_f = torch.tensor([itp, itp - cfg['time_scale']]).type(torch.float32).to(device)
        zero = torch.zeros(x.shape[0], 1).to(x)
        z, delta_logpx = model_f(x, zero, integration_times=integration_times_f)
        deltas_px.append(delta_logpx)
        
        # transform to previous timepoint
        
        # Straightline regularization
        # Integrate to random point at time t and assert close to (1 - t) * end + t * start
        #if args.interp_reg:
        #    t = np.random.rand()
        #    int_t = torch.tensor([itp - t * args.time_scale, itp])
        #    int_t = int_t.type(torch.float32).to(device)
        #    int_x = model_f(x, integration_times=int_t)
        #    int_x = int_x.detach()
        #    actual_int_x = x * (1 - t) + z * t # 直線状にあると考える.
        #    interp_loss += F.mse_loss(int_x, actual_int_x)
    
    #if args.interp_reg:
    #    print("interp_loss", interp_loss)

    # 標準正規分布の尤度
    logpz = data.base_density()(z)

    # build growth rates
    # if args.use_growth:
    #    growthrates = [torch.ones_like(logpz)]
    #    for z_state, tp in zip(zs[::-1], args.timepoints[:-1]):
    #        # Full state includes time parameter to growth_model
    #        time_state = tp * torch.ones(z_state.shape[0], 1).to(z_state)
    #        full_state = torch.cat([z_state, time_state], 1)
    #        growthrates.append(growth_model(full_state))

    # Accumulate losses
    losses = []
    logps = [logpz]
    rev_deltas_pypx = deltas_pypx[::-1]
    rev_deltas_px = deltas_px[::-1]
    for i, (delta_logpypx, delta_logpx) in enumerate(zip(rev_deltas_pypx, rev_deltas_px)):
        logpx = logps[-1] - delta_logpx

        # if args.use_growth:
        #     logpx += torch.log(torch.clamp(growthrates[i], 1e-4, 1e4))
        logps.append(logpx[: -cfg['batch_size']])

        logpy = logpx[-cfg['batch_size'] :] - delta_logpypx
        losses.append(-torch.mean(logpy))
        
    losses = torch.stack(losses)
    weights = torch.ones_like(losses).to(logpx)
    if cfg['leaveout_timepoint'] >= 0:
        weights[cfg['leaveout_timepoint']] = 0
    losses = torch.mean(losses * weights)

    # Direction regularization
    # (dy/dt) = (dy/dx) (dx/dt)
    # if args.vecint:
    #    similarity_loss = 0
    #    for i, (itp, tp) in enumerate(zip(args.int_tps, args.timepoints)):
    #        itp = torch.tensor(itp).type(torch.float32).to(device)
    #        idx = args.data.sample_index(args.batch_size, tp)
    #        x = args.data.get_data()[idx]
    #        v = args.data.get_velocity()[idx]
    #        x = torch.from_numpy(x).type(torch.float32).to(device)
    #        v = torch.from_numpy(v).type(torch.float32).to(device)
    #        x += torch.randn_like(x) * 0.1
    #        # Only penalizes at the time / place of visible samples
    #        direction = -model.chain[0].odefunc.odefunc.diffeq(itp, x)
    #        if args.use_magnitude:
    #            similarity_loss += torch.mean(F.mse_loss(direction, v))
    #        else:
    #            similarity_loss -= torch.mean(F.cosine_similarity(direction, v))
    #    logger.info(similarity_loss)
    #    losses += similarity_loss * args.vecint

    # losses += interp_loss
    return losses


def train(cfg, data, model_g, model_f, growth_model, regularization_coeffs_g, regularization_fns_g, 
    regularization_coeffs_f, regularization_fns_f, logger, out_dir, device
):
    optimizer_g = optim.Adam(
        model_g.parameters(), lr=cfg['model_g']['optim']['lr'], weight_decay=cfg['model_g']['optim']['weight_decay']
    )
    optimizer_f = optim.Adam(
        model_f.parameters(), lr=cfg['model_f']['optim']['lr'], weight_decay=cfg['model_f']['optim']['weight_decay']
    )

    nfef_meter_g = utils.RunningAverageMeter(0.93)
    nfeb_meter_g = utils.RunningAverageMeter(0.93)
    tt_meter_g = utils.RunningAverageMeter(0.93)

    
    nfef_meter_f = utils.RunningAverageMeter(0.93)
    nfeb_meter_f = utils.RunningAverageMeter(0.93)
    tt_meter_f = utils.RunningAverageMeter(0.93)

    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)

    best_loss = float("inf")
    if cfg['use_growth']:
        growth_model.eval()
    end = time.time()
    for itr in range(1, cfg['niters'] + 1):
        model_g.train()
        optimizer_g.zero_grad()
        model_f.train()
        optimizer_f.zero_grad()

        # Train
        if cfg['model_g']['reg']['spectral_norm']:
            spectral_norm_power_iteration(model_g, 1)

        if cfg['model_f']['reg']['spectral_norm']:
            spectral_norm_power_iteration(model_f, 1)

        loss = compute_loss(cfg, data, model_g, model_f, growth_model, logger, device)
        loss_meter.update(loss.item())

        if len(regularization_coeffs_g) > 0:
            # Only regularize on the last timepoint
            reg_states_g = get_regularization(model_g, regularization_coeffs_g)
            reg_loss_g = sum(
                reg_state * coeff
                for reg_state, coeff in zip(reg_states_g, regularization_coeffs_g)
                if coeff != 0
            )
            loss = loss + reg_loss_g

        if len(regularization_coeffs_f) > 0:
            reg_states_f = get_regularization(model_f, regularization_coeffs_f)
            reg_loss_f = sum(
                reg_state * coeff
                for reg_state, coeff in zip(reg_states_f, regularization_coeffs_f)
                if coeff != 0
            )
            loss = loss + reg_loss_f

        total_time_g = count_total_time(model_g)
        nfe_forward_g = count_nfe(model_g)

        total_time_f = count_total_time(model_f)
        nfe_forward_f = count_nfe(model_f)

        loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        # Eval
        nfe_total_g= count_nfe(model_g)
        nfe_backward = nfe_total_g - nfe_forward_g
        nfef_meter_g.update(nfe_forward_g)
        nfeb_meter_g.update(nfe_backward)
        tt_meter_g.update(total_time_g)

        nfe_total_f = count_nfe(model_f)
        nfe_backward = nfe_total_f - nfe_forward_f
        nfef_meter_f.update(nfe_forward_f)
        nfeb_meter_f.update(nfe_backward)
        tt_meter_f.update(total_time_f)

        time_meter.update(time.time() - end)

        """
        log_message = (
            "Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) |"
            " NFE Forward (g) {:.0f}({:.1f})"
            " | NFE Backward (g) {:.0f}({:.1f}) |"
            " NFE Forward (f) {:.0f}({:.1f})"
            " | NFE Backward (f) {:.0f}({:.1f})".format(
                itr,
                time_meter.val,
                time_meter.avg,
                loss_meter.val,
                loss_meter.avg,
                nfef_meter_g.val,
                nfef_meter_g.avg,
                nfeb_meter_g.val,
                nfeb_meter_g.avg,
                nfef_meter_f.val,
                nfef_meter_f.avg,
                nfeb_meter_f.val,
                nfeb_meter_f.avg,
            )
        )
        """
        log_message = (
            "Iter {:04d} | Loss {:.6f}({:.6f}) |"
            .format(
                itr,
                loss_meter.val,
                loss_meter.avg
            )
        )

        if len(regularization_coeffs_g) > 0:
            log_message = append_regularization_to_log(
                log_message, regularization_fns_g, reg_states_g, 'g'
            )

        if len(regularization_coeffs_f) > 0:
            log_message = append_regularization_to_log(
                log_message, regularization_fns_f, reg_states_f, 'f'
            )

        logger.info(log_message)

        if itr % cfg['val_freq'] == 0 or itr == cfg['niters']:
            with torch.no_grad():
                train_eval(
                    cfg, data, model_g, model_f, growth_model, itr, best_loss, logger, out_dir, device
                )

        if itr % cfg['viz_freq'] == 0:
            if data.get_shape()[0] > 2:
                logger.warning("Skipping vis as data dimension is >2")
            else:
                with torch.no_grad():
                    visualize(cfg, data, model_g, model_f, itr, out_dir, device)
        if itr % cfg['save_freq'] == 0:
            chkpt = {
                "state_dict_g": model_g.state_dict(),
                "state_dict_f": model_f.state_dict(),
            }
            if cfg['use_growth']:
                chkpt.update({"growth_state_dict": growth_model.state_dict()})

            utils.save_checkpoint(
                chkpt, out_dir, epoch=itr,
            )
        end = time.time()
    logger.info("Training has finished.")

def train_eval(cfg, data, model_g, model_f, growth_model, itr, best_loss, logger, out_dir, device):
    model_g.eval()
    model_f.eval()
    test_loss = compute_loss(cfg, data, model_g, model_f, growth_model, logger, device)
    test_nfe_g = count_nfe(model_g)
    test_nfe_f = count_nfe(model_f)

    log_message = "[TEST] Iter {:04d} | Test Loss {:.6f} |" " NFE (g) {:.0f} |" " NFE (f) {:.0f}".format(
        itr, test_loss, test_nfe_g, test_nfe_f
    )
    logger.info(log_message)
    utils.makedirs(out_dir)
    with open(os.path.join(out_dir, "train_eval.csv"), "a") as f:
        import csv

        writer = csv.writer(f)
        writer.writerow((itr, test_loss))

    if test_loss.item() < best_loss:
        best_loss = test_loss.item()
        chkpt = {
            "state_dict_g": model_g.state_dict(),
            "state_dict_f": model_f.state_dict(),
        }
        if cfg['use_growth']:
            chkpt.update({"growth_state_dict": growth_model.state_dict()})
        torch.save(
            chkpt, os.path.join(out_dir, "checkpt.pth"),
        )


def visualize(cfg, data, model_g, model_f, itr, out_dir, device):
    model_g.eval()
    model_f.eval()
    for i, tp in enumerate(cfg['timepoints']):
        idx = data.sample_index(cfg['viz_batch_size'], tp)
        p_samples = data.get_data()[idx]
        sample_fn, density_fn = get_transforms(
            cfg, model_g, model_f, cfg['int_tps'][: i + 1], device
        )
        plt.figure(figsize=(9, 3))
        visualize_transform(
            p_samples,
            data.base_sample(),
            data.base_density(),
            transform=sample_fn,
            inverse_transform=density_fn,
            samples=True,
            npts=100,
            device=device,
        )
        fig_filename = os.path.join(
            out_dir, "figs", "{:04d}_{:01d}.jpg".format(itr, i)
        )
        utils.makedirs(os.path.dirname(fig_filename))
        plt.savefig(fig_filename)
        plt.close()


def plot_output(cfg, data, model_g, model_f, out_dir, device):
    save_traj_dir = os.path.join(out_dir, "trajectory")
    # logger.info('Plotting trajectory to {}'.format(save_traj_dir))
    data_samples = data.get_data()[data.sample_index(2000, 0)]
    np.random.seed(42)
    start_points = data.base_sample()(1000, 2)
    # idx = args.data.sample_index(50, 0)
    # start_points = args.data.get_data()[idx]
    # start_points = torch.from_numpy(start_points).type(torch.float32)
    save_vectors(
        data.base_density(),
        model_g,
        model_f,
        start_points,
        data.get_data(),
        data.get_times(),
        out_dir,
        skip_first=(not data.known_base_density()),
        device=device,
        end_times=cfg['int_tps'],
        ntimes=100,
    )
    save_trajectory(
        data.base_density(),
        data.base_sample(),
        model_g,
        model_f,
        data_samples,
        save_traj_dir,
        device=device,
        end_times=cfg['int_tps'],
        ntimes=25,
    )
    # trajectory_to_video(save_traj_dir)

    density_dir = os.path.join(out_dir, "density2")
    save_trajectory_density(
        data.base_density(),
        model_g,
        model_f,
        data_samples,
        density_dir,
        device=device,
        end_times=cfg['int_tps'],
        ntimes=25,
        memory=0.1,
    )
    # trajectory_to_video(density_dir)


def main(args):
    # logger
    print(args.no_display_loss)
    utils.makedirs(args.out_dir)
    logger = utils.get_logger(
        logpath=os.path.join(args.out_dir, "logs"),
        filepath=os.path.abspath(__file__),
        displaying=~args.no_display_loss,
    )

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    with open(Path(args.out_dir) / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=4)

    if cfg['model_g']['layer_type'] == "blend":
        logger.info("!! Setting time_scale from None to 1.0 for Blend layers.")
        cfg['model_g']['time_scale'] = 1.0

    if cfg['model_f']['layer_type'] == "blend":
        logger.info("!! Setting time_scale from None to 1.0 for Blend layers.")
        cfg['model_f']['time_scale'] = 1.0

    logger.info(cfg)

    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    if args.use_cpu:
        device = torch.device("cpu")

    data = dataset.SCData.factory(cfg['dataset'])

    # タイムポイントが得られている時間 ( e.g. [0, 1, 2, 3] )
    timepoints = data.get_unique_times()
    cfg['timepoints'] = timepoints
    # Use maximum timepoint to establish integration_times
    # as some timepoints may be left out for validation etc.
    # 時刻0: ランダムノイズ, 時刻 time_scale: 初期状態(t=0), 時刻 k * time_scale: 状態(k=t+1)
    
    cfg['int_tps'] = (np.arange(max(timepoints) + 1) + 1.0) * cfg['time_scale']

    # regularization_fns: regularization関数, regularization_coeffs: regularizationの係数
    regularization_fns_g, regularization_coeffs_g = create_regularization_fns(cfg['model_g']['reg'])
    regularization_fns_f, regularization_coeffs_f = create_regularization_fns(cfg['model_f']['reg'])

    model_g = build_model_tabular(cfg['model_g'], data.get_shape()[0], regularization_fns_g).to(device)
    model_f = build_model_tabular(cfg['model_f'], data.get_shape()[0], regularization_fns_f).to(device)

    growth_model = None
    if cfg['use_growth']:
        if cfg['leaveout_timepoint'] == -1:
            growth_model_path = "../data/externel/growth_model_v2.ckpt"
        elif cfg['leaveout_timepoint'] in [1, 2, 3]:
            assert cfg['dataset']['max_dim'] == 5
            growth_model_path = "../data/growth/model_%d" % cfg['leaveout_timepoint']
        else:
            print("WARNING: Cannot use growth with this timepoint")

        growth_model = torch.load(growth_model_path, map_location=device)
    # spectral normalization
    if cfg['model_g']['reg']['spectral_norm']:
        add_spectral_norm(model_g)
    
    if cfg['model_f']['reg']['spectral_norm']:
        add_spectral_norm(model_f)

    set_cnf_options(cfg['model_g'], model_g)
    set_cnf_options(cfg['model_f'], model_f)

    logger.info(model_g)
    logger.info(model_f)
    n_param_g = count_parameters(model_g)
    n_param_f = count_parameters(model_f)
    logger.info("Number of trainable parameters: g: {}, f: {}".format(n_param_g, n_param_f))

    train(
        cfg,
        data,
        model_g,
        model_f,
        growth_model,
        regularization_coeffs_g,
        regularization_fns_g,
        regularization_coeffs_f,
        regularization_fns_f,
        logger,
        args.out_dir,
        device
    )

    if data.data.shape[1] == 2:
        plot_output(cfg, data, model_g, model_f, args.out_dir, device)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Continuous Normalizing Flow")
    parser.add_argument('-config', '-c', help="the configuration file for training.", type=str, required=True)
    parser.add_argument("-out_dir", '-o', help="the directory to output the analysis results.", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--no_display_loss", action="store_false")
    args = parser.parse_args()
    
    main(args)
