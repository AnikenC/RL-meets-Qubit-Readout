import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import copy

import chex

from typing import Optional

from envs.single_photon_env import SinglePhotonLangevinReadoutEnv
from rl_algos.rl_wrappers import VecEnv


def stability_tester(
    key: chex.PRNGKey,
    env: SinglePhotonLangevinReadoutEnv,
    waveforms: jnp.ndarray,
    err: float,
):
    ### Process Waveforms ###
    # Scale Amplitudes appropriately as would be done in experiment
    waveforms_scaled = waveform_err_scaler(env, err, waveforms)

    # No need to scale durations if fixed sim_t1 is used

    # Run Simulations
    vec_env = VecEnv(env)
    rng, _rng = jax.random.split(key)
    rng_reset = jax.random.split(_rng, len(waveforms))

    _, vec_init_state = vec_env.reset(rng_reset, None)

    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, len(waveforms))

    obs, state, reward, done, info = vec_env.step(
        rng_step, vec_init_state, waveforms_scaled, None
    )

    max_pFs = state.max_pf
    photon_times = state.photon_time
    max_photons = state.max_photon

    return max_pFs, photon_times, max_photons


def get_kc_arrs(kappa: float, chi: float, err: float):
    err_arr = jnp.array([1.0 - err, 1.0, 1.0 + err])
    k_arr = kappa * err_arr
    c_arr = chi * err_arr
    kc_grid = jnp.array(jnp.meshgrid(k_arr, c_arr)).reshape(2, -1).T
    kappa_vals = kc_grid[:, 0]
    chi_vals = kc_grid[:, 1]
    return k_arr, c_arr, kappa_vals, chi_vals


def waveform_err_scaler(
    env: SinglePhotonLangevinReadoutEnv, err: float, waveforms: jnp.ndarray
):
    kappa = env._kappa
    chi = env._chi
    _, _, k_vals, c_vals = get_kc_arrs(kappa, chi, err)

    n0 = 4 * env.a0**2 / (kappa**2 + chi**2)
    a0_vals = 0.5 * jnp.sqrt(n0 * (k_vals**2 + c_vals**2))
    waveforms_scaled = waveforms / a0_vals.reshape(-1, 1)
    return waveforms_scaled


def waveform_kappa_chi_stability_tester(
    key: chex.PRNGKey,
    waveform: jnp.ndarray,
    env_name: str,
    main_env_config: dict,
    error_percentage: Optional[float] = 10.0,
    num_vals: Optional[int] = 5,
):
    reduced_config = copy.deepcopy(main_env_config)
    kappa = reduced_config.pop("kappa")
    chi = reduced_config.pop("chi")

    env_class = SinglePhotonLangevinReadoutEnv

    def env_tester(key: chex.PRNGKey, kappa: float, chi: float):
        env = env_class(kappa=kappa, chi=chi, **reduced_config)
        env.shot_noise_std = 0.0

        rng, _rng = jax.random.split(key)
        params = env.default_params
        init_obs, init_state = env.reset(_rng, params)

        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, init_state, waveform, params)

        max_pF = state.max_pf
        max_photon = state.max_photon
        photon_time = state.photon_time

        return max_pF, max_photon, photon_time

    jitted_tester = jax.jit(jax.vmap(env_tester))

    error_val = error_percentage / 100.0
    kappas = jnp.linspace(
        kappa * (1.0 - error_val), kappa * (1.0 + error_val), num_vals
    )
    chis = jnp.linspace(chi * (1.0 - error_val), chi * (1.0 + error_val), num_vals)

    kappa_chi_grid = jnp.array(jnp.meshgrid(kappas, chis)).reshape(2, -1).T
    kappa_grid = kappa_chi_grid[:, 0]
    chi_grid = kappa_chi_grid[:, 1]

    rng, _rng = jax.random.split(key)
    rng_tester = jax.random.split(_rng, kappa_chi_grid.shape[0])

    res_pF, res_photon, res_photon_time = jitted_tester(
        rng_tester, kappa_grid, chi_grid
    )

    res_pF = res_pF.reshape(num_vals, -1)
    res_photon = res_photon.reshape(num_vals, -1)
    res_photon_time = res_photon_time.reshape(num_vals, -1)
    return res_pF, res_photon, res_photon_time, kappa_chi_grid, kappas, chis


def plot_learning(
    rewards: jnp.ndarray,
    max_pFs: jnp.ndarray,
    photon_times: jnp.ndarray,
    smoothnesses: jnp.ndarray,
    bandwidths: jnp.ndarray,
):
    """
    Takes in arrays of shape (num_updates, num_envs)
    """
    num_updates = rewards.shape[0]

    mean_rewards = jnp.mean(rewards, axis=-1)
    std_rewards = jnp.std(rewards, axis=-1)

    mean_pFs = jnp.mean(max_pFs, axis=-1)
    std_pFs = jnp.std(max_pFs, axis=-1)

    mean_times = jnp.mean(photon_times, axis=-1)
    std_times = jnp.std(photon_times, axis=-1)

    mean_smoothnesses = jnp.mean(smoothnesses, axis=-1)
    std_smoothnesses = jnp.std(smoothnesses, axis=-1)

    mean_bandwidths = jnp.mean(bandwidths, axis=-1)
    std_bandwidths = jnp.std(bandwidths, axis=-1)

    fig, ax = plt.subplots(5, figsize=(8.0, 16.0))

    updates = jnp.arange(num_updates)

    ax[0].plot(mean_rewards, label="mean rewards")
    ax[0].fill_between(
        updates,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        color="orange",
        alpha=0.4,
    )
    ax[0].set_xlabel("Updates")
    ax[0].set_ylabel("Mean Batch Reward")
    ax[0].legend()

    ax[1].plot(mean_pFs, label="mean pFs")
    ax[1].fill_between(
        updates,
        mean_pFs - std_pFs,
        mean_pFs + std_pFs,
        color="orange",
        alpha=0.4,
    )
    ax[1].set_xlabel("Updates")
    ax[1].set_ylabel("Mean Batch pF")
    ax[1].legend()

    ax[2].plot(mean_times, label="mean photon times")
    ax[2].fill_between(
        updates,
        mean_times - std_times,
        mean_times + std_times,
        color="orange",
        alpha=0.4,
    )
    ax[2].set_xlabel("Updates")
    ax[2].set_ylabel("Mean Batch Photon Time")
    ax[2].legend()

    ax[3].plot(mean_smoothnesses, label="mean noisiness")
    ax[3].fill_between(
        updates,
        mean_smoothnesses - std_smoothnesses,
        mean_smoothnesses + std_smoothnesses,
        color="orange",
        alpha=0.4,
    )
    ax[3].set_xlabel("Updates")
    ax[3].set_ylabel("Mean Batch Noisiness")
    ax[3].set_yscale("log")
    ax[3].legend()

    ax[4].plot(mean_bandwidths, label="mean bandwidths")
    ax[4].fill_between(
        updates,
        mean_bandwidths - std_bandwidths,
        mean_bandwidths + std_bandwidths,
        color="orange",
        alpha=0.4,
    )
    ax[4].set_xlabel("Updates")
    ax[4].set_ylabel("Mean Batch Bandwidths")
    ax[4].legend()

    plt.show()
