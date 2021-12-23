import six
import math

import lib.layers.wrappers.cnf_regularization as reg_lib
import lib.spectral_norm as spectral_norm
import lib.layers as layers
from lib.layers.odefunc import divergence_bf, divergence_approx


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def set_cnf_options(cfg, model):
    cfg_solver = cfg['ode_solver']
    def _set(module):
        if isinstance(module, layers.CNF):
            # Set training settings
            
            module.solver = cfg_solver['solver']
            module.atol = cfg_solver['atol']
            module.rtol = cfg_solver['rtol']
            if cfg_solver['step_size'] is not None:
                module.solver_options["step_size"] = cfg_solver['step_size']

            # If using fixed-grid adams, restrict order to not be too high.
            if cfg_solver['solver'] in ["fixed_adams", "explicit_adams"]:
                module.solver_options["max_order"] = 4

            # Set the test settings
            module.test_solver = cfg_solver['test_solver'] if cfg_solver['test_solver'] else cfg_solver['solver']
            module.test_atol = cfg_solver['test_atol'] if cfg_solver['test_atol'] else cfg_solver['atol']
            module.test_rtol = cfg_solver['test_rtol'] if cfg_solver['test_rtol'] else cfg_solver['rtol']

        if isinstance(module, layers.ODEfunc):
            module.rademacher = cfg['rademacher']
            module.residual = cfg['residual']

    model.apply(_set)


def override_divergence_fn(model, divergence_fn):
    def _set(module):
        if isinstance(module, layers.ODEfunc):
            if divergence_fn == "brute_force":
                module.divergence_fn = divergence_bf
            elif divergence_fn == "approximate":
                module.divergence_fn = divergence_approx

    model.apply(_set)


def count_nfe(model):
    class AccNumEvals(object):
        def __init__(self):
            self.num_evals = 0

        def __call__(self, module):
            if isinstance(module, layers.CNF):
                self.num_evals += module.num_evals()

    accumulator = AccNumEvals()
    model.apply(accumulator)
    return accumulator.num_evals


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_time(model):
    class Accumulator(object):
        def __init__(self):
            self.total_time = 0

        def __call__(self, module):
            if isinstance(module, layers.CNF):
                self.total_time = (
                    self.total_time + module.sqrt_end_time * module.sqrt_end_time
                )

    accumulator = Accumulator()
    model.apply(accumulator)
    return accumulator.total_time


def add_spectral_norm(model, logger=None):
    """Applies spectral norm to all modules within the scope of a CNF."""

    def apply_spectral_norm(module):
        if "weight" in module._parameters:
            if logger:
                logger.info("Adding spectral norm to {}".format(module))
            spectral_norm.inplace_spectral_norm(module, "weight")

    def find_cnf(module):
        if isinstance(module, layers.CNF):
            module.apply(apply_spectral_norm)
        else:
            for child in module.children():
                find_cnf(child)

    find_cnf(model)


def spectral_norm_power_iteration(model, n_power_iterations=1):
    def recursive_power_iteration(module):
        if hasattr(module, spectral_norm.POWER_ITERATION_FN):
            getattr(module, spectral_norm.POWER_ITERATION_FN)(n_power_iterations)

    model.apply(recursive_power_iteration)


REGULARIZATION_FNS = {
    "l1int": reg_lib.l1_regularzation_fn,
    "l2int": reg_lib.l2_regularzation_fn,
    "sl2int": reg_lib.squared_l2_regularization_fn,
    "dl2int": reg_lib.directional_l2_regularization_fn,
    "dtl2int": reg_lib.directional_l2_change_penalty_fn,
    "JFrobint": reg_lib.jacobian_frobenius_regularization_fn,
    "JdiagFrobint": reg_lib.jacobian_diag_frobenius_regularization_fn,
    "JoffdiagFrobint": reg_lib.jacobian_offdiag_frobenius_regularization_fn,
}

INV_REGULARIZATION_FNS = {v: k for k, v in six.iteritems(REGULARIZATION_FNS)}


def append_regularization_to_log(log_message, regularization_fns, reg_states, model_name):
    for i, reg_fn in enumerate(regularization_fns):
        log_message = (
            log_message
            + " | "
            + f"{model_name}: "
            + INV_REGULARIZATION_FNS[reg_fn]
            + ": {:.8f}".format(reg_states[i].item())
        )
    return log_message


def create_regularization_fns(cfg):
    regularization_fns = []
    regularization_coeffs = []

    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        if cfg[arg_key] is not None:
            regularization_fns.append(reg_fn)
            regularization_coeffs.append(cfg[arg_key])

    regularization_fns = tuple(regularization_fns)
    regularization_coeffs = tuple(regularization_coeffs)
    return regularization_fns, regularization_coeffs


def get_regularization(model, regularization_coeffs):
    if len(regularization_coeffs) == 0:
        return None
    acc_reg_states = tuple([0.0] * len(regularization_coeffs))
    for module in model.modules():
        if isinstance(module, layers.CNF):
            acc_reg_states = tuple(
                acc + reg
                for acc, reg in zip(acc_reg_states, module.get_regularization_states())
            )
    return acc_reg_states


def build_model_tabular(cfg, dims, regularization_fns=None):
    hidden_dims = tuple(map(int, cfg['dims'].split("-")))

    def build_cnf():
        diffeq = layers.ODEnet(
            hidden_dims=hidden_dims,
            input_shape=(dims,),
            strides=None,
            conv=False,
            layer_type=cfg['layer_type'],
            nonlinearity=cfg['nonlinearity'],
        )
        odefunc = layers.ODEfunc(
            diffeq=diffeq,
            divergence_fn=cfg['divergence_fn'],
            residual=cfg['residual'],
            rademacher=cfg['rademacher'],
        )
        cnf = layers.CNF(
            odefunc=odefunc,
            T=cfg['time_scale'],
            train_T=cfg['train_T'],
            regularization_fns=regularization_fns,
            solver=cfg['ode_solver']['solver'],
        )
        return cnf

    chain = [build_cnf() for _ in range(cfg['num_blocks'])]
    if cfg['reg']['batch_norm']:
        bn_layers = [
            layers.MovingBatchNorm1d(dims, bn_lag=cfg['reg']['bn_lag'])
            for _ in range(cfg.num_blocks)
        ]
        bn_chain = [layers.MovingBatchNorm1d(dims, bn_lag=cfg['reg']['bn_lag'])]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = layers.SequentialFlow(chain)

    set_cnf_options(cfg, model)
    return model
