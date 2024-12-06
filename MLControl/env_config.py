import copy
from typing import Optional
import jax.numpy as jnp

### Parameters collected from IBMQ Backends with experiments run in Qiskit

# Kyoto Q2 Sim Config
_kyoto_sim_config = {
    "kappa": 10.07,
    "chi": 0.92 * 2.0 * jnp.pi,
    "kerr": 0.002,
    "time_coeff": 2.0,
    "snr_coeff": 20.0,
    "smoothness_coeff": 1.0,
    "smoothness_baseline_scale": 0.5,
    "gauss_kernel_len": 15,
    "gauss_kernel_std": 2.0,
    "bandwidth": 50.0,
    "freq_relative_cutoff": 0.1,
    "bandwidth_coeff": 0.0,
    "n0": 25.5,
    "tau_0": 0.783,
    "res_amp_scaling": 1 / 0.51,
    "nR": 0.1,
    "snr_scale_factor": 0.5513744339600395,
    "gamma_I": 1 / 95.43154150750762,
    "photon_gamma": 0.0,
    "sim_t1": 0.7,
    "init_fid": 0.9905278732070423,
    "photon_weight": 8.0,
    "standard_fid": 0.99,
    "shot_noise_std": 0.0,
    "max_grad": 38,
}

# Brisbane Q2 Sim Config
_brisbane_sim_config = {
    "kappa": 21.44,
    "chi": 0.305 * 2.0 * jnp.pi,
    "kerr": 0.002,
    "time_coeff": 2.0,
    "snr_coeff": 20.0,
    "smoothness_coeff": 1.0,
    "smoothness_baseline_scale": 0.5,
    "gauss_kernel_len": 15,
    "gauss_kernel_std": 2.0,
    "bandwidth": 50.0,
    "freq_relative_cutoff": 0.1,
    "bandwidth_coeff": 0.0,
    "n0": 58.55,
    "tau_0": 0.720,
    "res_amp_scaling": 1 / 0.279,
    "nR": 0.1,
    "snr_scale_factor": 1.3215068094431675,
    "gamma_I": 1 / 667.4744288471769,
    "photon_gamma": 1 / 5280.129954339075,
    "sim_t1": 0.7,
    "init_fid": 1.0,
    "photon_weight": 8.0,
    "standard_fid": 1.0,
    "shot_noise_std": 0.0,
    "max_grad": 38,
}


def get_kyoto_config():
    return _kyoto_sim_config


def get_brisbane_config():
    return _brisbane_sim_config
