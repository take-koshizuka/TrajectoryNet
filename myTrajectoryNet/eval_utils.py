import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from optimal_transport.emd import earth_mover_distance


def generate_samples(device, args, model, growth_model, n=10000, timepoint=None):
    """ generates samples using model and base density

    This is useful for measuring the wasserstein distance between the
    predicted distribution and the true distribution for evaluation
    purposes against other types of models. We should use
    negative log likelihood if possible as it is deterministic and
    more discriminative for this model type.

    TODO: Is this biased???
    """
    z_samples = args.data.base_sample()(n, *args.data.get_shape()).to(device)
    # Forward pass through the model / growth model
    with torch.no_grad():
        int_list = [
            torch.tensor([it - args.time_scale, it]).type(torch.float32).to(device)
            for it in args.int_tps[: timepoint + 1]
        ]

        logpz = args.data.base_density()(z_samples)
        z = z_samples
        for it in int_list:
            z, logpz = model(z, logpz, integration_times=it, reverse=True)
        z = z.cpu().numpy()
        np.save(os.path.join(args.save, "samples_%0.2f.npy" % timepoint), z)
        logpz = logpz.cpu().numpy()
        plt.scatter(z[:, 0], z[:, 1], s=0.1, alpha=0.5)
        original_data = args.data.get_data()[args.data.get_times() == timepoint]
        idx = np.random.randint(original_data.shape[0], size=n)
        samples = original_data[idx, :]
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
        plt.savefig(os.path.join(args.save, "samples%d.png" % timepoint))
        plt.close()

        pz = np.exp(logpz)
        pz = pz / np.sum(pz)
        print(pz)

        print(
            earth_mover_distance(
                original_data, samples + np.random.randn(*samples.shape) * 0.1
            )
        )

        print(earth_mover_distance(z, original_data))
        print(earth_mover_distance(z, samples))
        # print(earth_mover_distance(z, original_data, weights1=pz.flatten()))
        # print(
        #    earth_mover_distance(
        #        args.data.get_data()[args.data.get_times() == (timepoint - 1)],
        #        original_data,
        #    )
        # )

    if args.use_growth and growth_model is not None:
        raise NotImplementedError(
            "generating samples with growth model is not yet implemented"
        )


def calculate_path_length(device, args, model, data, end_time, n_pts=10000):
    """ Calculates the total length of the path from time 0 to timepoint
    """
    # z_samples = torch.tensor(data.get_data()).type(torch.float32).to(device)
    z_samples = data.base_sample()(n_pts, *data.get_shape()).to(device)
    model.eval()
    n = 1001
    with torch.no_grad():
        integration_times = (
            torch.tensor(np.linspace(0, end_time, n)).type(torch.float32).to(device)
        )
        # z, _ = model(z_samples, torch.zeros_like(z_samples), integration_times=integration_times, reverse=False)
        z, _ = model(
            z_samples,
            torch.zeros_like(z_samples),
            integration_times=integration_times,
            reverse=True,
        )
        z = z.cpu().numpy()
        z_diff = np.diff(z, axis=0)
        z_lengths = np.sum(np.linalg.norm(z_diff, axis=-1), axis=0)
        total_length = np.mean(z_lengths)
        import ot as pot
        from scipy.spatial.distance import cdist

        emd = pot.emd2(
            np.ones(n_pts) / n_pts,
            np.ones(n_pts) / n_pts,
            cdist(z[-1, :, :], data.get_data()),
        )
        print(total_length, emd)
        plt.scatter(z[-1, :, 0], z[-1, :, 1])
        plt.savefig("test.png")
        plt.close()


def evaluate_mse(device, args, model_g, model_f, growth_model=None):
    if args.use_growth or growth_model is not None:
        print("WARNING: Ignoring growth model and computing anyway")

    paths = args.data.get_paths()

    z_samples = torch.tensor(paths[:, 0, :]).type(torch.float32).to(device)
    # Forward pass through the model / growth model
    with torch.no_grad():
        int_list = [
            torch.tensor([it - args.time_scale, it]).type(torch.float32).to(device)
            for it in args.int_tps
        ]

        logpz = args.data.base_density()(z_samples)
        z = z_samples
        ys = []
        for it in int_list:
            z, logpz = model_f(z, logpz, integration_times=it, reverse=True)
            y = model_g(z, reverse=True)
            ys.append(y.cpu().numpy())
        ys = np.stack(ys)
        np.save(os.path.join(args.save, "path_samples.npy"), ys)

        # logpz = logpz.cpu().numpy()
        # plt.scatter(z[:, 0], z[:, 1], s=0.1, alpha=0.5)
        mses = []
        print(ys.shape, paths[:, 1, :].shape)
        for tpi in range(len(args.timepoints)):
            mses.append(np.mean((paths[:, tpi + 1, :] - ys[tpi]) ** 2, axis=(-2, -1)))
        mses = np.array(mses)
        print(mses.shape)
        np.save(os.path.join(args.save, "mses.npy"), mses)
        return mses


def evaluate_kantorovich_v2(device, cfg, data, model_g, model_f, growth_model=None):
    """ Eval the model via kantorovich distance on leftout timepoint

    v2 computes samples from subsequent timepoint instead of base distribution.
    this is arguably a fairer comparison to other methods such as WOT which are
    not model based this should accumulate much less numerical error in the
    integration procedure. However fixes to the number of samples to the number in the
    previous timepoint.

    If we have a growth model we should use this to modify the weighting of the
    points over time.
    """
    if cfg['use_growth'] or growth_model is not None:
        # raise NotImplementedError(
        #    "generating samples with growth model is not yet implemented"
        # )
        print("WARNING: Ignoring growth model and computing anyway")

    # Backward pass through the model / growth model
    with torch.no_grad():
        integration_times_g =  torch.tensor([0.0, cfg['model_g']['time_scale']])
        integration_times_f = torch.tensor(
            [
                cfg['int_tps'][cfg['leaveout_timepoint'] + 1],
                cfg['int_tps'][cfg['leaveout_timepoint']],
            ]
        ).type(torch.float32).to(device)
        next_y = data.get_data('all')[
            data.get_times() == cfg['leaveout_timepoint'] + 1
        ]
        next_y = torch.from_numpy(next_y).type(torch.float32).to(device)
        
        next_x = model_g(next_y, integration_times=integration_times_g, reverse=False)

        zero = torch.zeros(next_x.shape[0], 1).to(device)
        x_backward, _ = model_f.chain[0](next_x, zero, integration_times=integration_times_f)
        y_backward = model_g(x_backward, integration_times=integration_times_g, reverse=True)
        y_backward = y_backward.cpu().numpy()


        prev_y = data.get_data('all')[
            data.get_times() == cfg['leaveout_timepoint'] - 1
        ]
        # 後ろから
        prev_y = torch.from_numpy(prev_y).type(torch.float32).to(device)
        integration_times_f = torch.tensor(
            [
                cfg['int_tps'][cfg['leaveout_timepoint']],
                cfg['int_tps'][cfg['leaveout_timepoint'] - 1],
            ]
        ).type(torch.float32).to(device)
        prev_x = model_g(prev_y, integration_times=integration_times_g, reverse=False)
        zero = torch.zeros(prev_x.shape[0], 1).to(device)
        x_forward, _ = model_f.chain[0](prev_x, zero, integration_times=integration_times_f, reverse=True)
        y_forward = model_g(x_forward, integration_times=integration_times_g, reverse=True)
        y_forward = y_forward.cpu().numpy()

        emds = []
        for tpi in [cfg['leaveout_timepoint']]:
            original_data = data.get_data('all')[
                data.get_times() == cfg['timepoints'][tpi]
            ]
            emds.append(earth_mover_distance(y_backward, original_data))
            emds.append(earth_mover_distance(y_forward, original_data))

        emds = np.array(emds)
        return emds


def evaluate_kantorovich(device, cfg, data, model_g, model_f, growth_model=None, n=10000):
    """ Eval the model via kantorovich distance on all timepoints

    compute samples forward from the starting parametric distribution keeping track
    of growth rate to scale the final distribution.

    The growth model is a single model of time independent cell growth /
    death rate defined as a variation from uniform.

    If we have a growth model we should use this to modify the weighting of the
    points over time.
    """
    if cfg['use_growth'] or growth_model is not None:
        # raise NotImplementedError(
        #    "generating samples with growth model is not yet implemented"
        # )
        print("WARNING: Ignoring growth model and computing anyway")

    z_samples = data.base_sample()(n, *data.get_shape()).to(device)
    # Forward pass through the model / growth model
    with torch.no_grad():
        int_list = []
        for i, it in enumerate(cfg['int_tps']):
            if i == 0:
                prev = 0.0
            else:
                prev = cfg['int_tps'][i - 1]
            int_list.append(torch.tensor([it, prev]).type(torch.float32).to(device))

        # int_list = [
        #    torch.tensor([it - args.time_scale, it]).type(torch.float32).to(device)
        #    for it in args.int_tps
        # ]
        print(cfg['int_tps'])

        # logpz = data.base_density()(z_samples)
        z = z_samples
        ys = []
        growthrates = [torch.ones(z_samples.shape[0], 1).to(device)]

        integration_times_g =  torch.tensor([0.0, cfg['model_g']['time_scale']])
        for it, tp in zip(int_list, cfg['timepoints']):
            z = model_f(z, integration_times=it, reverse=True)
            y = model_g(z, integration_times=integration_times_g, reverse=True)
            ys.append(y.cpu().numpy())
            #if args.use_growth:
            #    time_state = tp * torch.ones(z.shape[0], 1).to(device)
            #    full_state = torch.cat([z, time_state], 1)
            #    # Multiply growth rates together to get total mass along path
            #    growthrates.append(
            #        torch.clamp(growth_model(full_state), 1e-4, 1e4) * growthrates[-1]
            #    )
        ys = np.stack(ys)
        #if args.use_growth:
        #    growthrates = growthrates[1:]
        #    growthrates = torch.stack(growthrates)
        #    growthrates = growthrates.cpu().numpy()
        #    np.save(os.path.join(args.save, "sample_weights.npy"), growthrates)
        # np.save(os.path.join(cfg['save'], "samples.npy"), ys)

        # logpz = logpz.cpu().numpy()
        # plt.scatter(z[:, 0], z[:, 1], s=0.1, alpha=0.5)
        emds = []
        for tpi in range(len(cfg['timepoints'])):
            original_data = data.get_data('all')[
                data.get_times() == cfg['timepoints'][tpi]
            ]
            if cfg['use_growth']:
                emds.append(
                    earth_mover_distance(
                        ys[tpi], original_data, weights1=growthrates[tpi].flatten()
                    )
                )
            else:
                emds.append(earth_mover_distance(ys[tpi], original_data))

        # Add validation point kantorovich distance evaluation
        if data.has_validation_samples():
            for tpi in np.unique(data.val_labels):
                original_data = data.val_data[
                    data.val_labels == cfg['timepoints'][tpi]
                ]
                if cfg['use_growth']:
                    emds.append(
                        earth_mover_distance(
                            ys[tpi], original_data, weights1=growthrates[tpi].flatten()
                        )
                    )
                else:
                    emds.append(earth_mover_distance(ys[tpi], original_data))

        emds = np.array(emds)
        return ys, emds

def evaluate(device, cfg, data, model_g, model_f, growth_model=None):
    """ Eval the model via negative log likelihood on all timepoints

    Compute loss by integrating backwards from the last time step
    At each time step integrate back one time step, and concatenate that
    to samples of the empirical distribution at that previous timestep
    repeating over and over to calculate the likelihood of samples in
    later timepoints iteratively, making sure that the ODE is evaluated
    at every time step to calculate those later points.

    The growth model is a single model of time independent cell growth /
    death rate defined as a variation from uniform.
    """
    use_growth = args.use_growth and growth_model is not None

    deltas_pypx = []
    deltas_px = []
    zs = []
    z = None
    for i, (itp, tp) in enumerate(zip(args.int_tps[::-1], args.timepoints[::-1])):
        # tp counts down from last
        integration_times = torch.tensor([itp - args.time_scale, itp])
        integration_times = integration_times.type(torch.float32).to(device)
        # integration_times.requires_grad = True
        
        # x[t] -> x[t-1] -> ... 
        y = args.data.get_data()[args.data.get_times() == tp]
        y = torch.from_numpy(y).type(torch.float32).to(device)

        x, delta_logpypx = model_g(y, zero)
        deltas_pypx.append(delta_logpypx)

        if i > 0:
            x = torch.cat((z, x))
            zs.append(z)

        zero = torch.zeros(x.shape[0], 1).to(x)
        z, delta_logpx = model_f(x, zero, integration_times=integration_times)
        deltas_px.append(delta_logpx)
        
    logpz = args.data.base_density()(z)

    # build growth rates
    # if use_growth:
    #    growthrates = [torch.ones_like(logpz)]
    #    for z_state, tp in zip(zs[::-1], args.timepoints[::-1][1:]):
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
        tp = args.timepoints[i]
        n_cells_in_tp = np.sum(args.data.get_times() == tp)
        logpx = logps[-1] - delta_logpx
        # if args.use_growth:
        #     logpx += torch.log(torch.clamp(growthrates[i], 1e-4, 1e4))
        logps.append(logpx[: -n_cells_in_tp])

        logpy = logps[-1] - delta_logpypx
        losses.append(-torch.mean(logpy[-n_cells_in_tp:]))
        
    losses = torch.stack(losses).cpu().numpy()
    np.save(os.path.join(args.save, "nll.npy"), losses)
    return losses
