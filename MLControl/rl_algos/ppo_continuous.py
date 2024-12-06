"""Adapted from PureJaxRL"""

import jax
import jax.numpy as jnp
from jax import jit, config, vmap, block_until_ready
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import chex

from rl_algos.rl_wrappers import VecEnv

from envs.single_photon_env import SinglePhotonLangevinReadoutEnv

import time

import matplotlib.pyplot as plt


class SeparateActorCritic(nn.Module):
    """
    Actor and Critic with Separate Feed-forward Neural Networks
    """

    action_dim: Sequence[int]
    activation: str = "tanh"
    layer_size: int = 128

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        if self.activation == "elu":
            activation = nn.elu
        if self.activation == "leaky_relu":
            activation = nn.leaky_relu
        if self.activation == "relu6":
            activation = nn.relu6
        if self.activation == "selu":
            activation = nn.selu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            self.layer_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.layer_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            self.layer_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            self.layer_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class CombinedActorCritic(nn.Module):
    """
    Actor and Critic Class with combined Feed-forward Neural Network
    """

    action_dim: Sequence[int]
    activation: str = "tanh"
    layer_size: int = 128

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        if self.activation == "elu":
            activation = nn.elu
        if self.activation == "leaky_relu":
            activation = nn.leaky_relu
        if self.activation == "relu6":
            activation = nn.relu6
        if self.activation == "selu":
            activation = nn.selu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            self.layer_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.layer_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean_val = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean_val, jnp.exp(actor_logtstd))

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            actor_mean
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    """
    Class for carrying RL State between processes
    """

    # done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    # info: jnp.ndarray


def PPO_make_train(config):
    """
    Function that returns a trainable function for an input configuration dictionary
    """
    env = SinglePhotonLangevinReadoutEnv(**config["ENV_PARAMS"])
    env = VecEnv(env)

    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    env_params = None

    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )

    def train(
        rng: chex.PRNGKey,
        num_envs: int,
    ):
        """
        Training function for environment
        """
        # INIT NETWORK
        network = CombinedActorCritic(
            env.action_space(env_params).shape[0],
            activation=config["ACTIVATION"],
            layer_size=config["LAYER_SIZE"],
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num_envs)
        obsv, env_state = env.reset(reset_rng, env_params)

        # We only use init_x as the batched observation, and env_state for state information

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # CALCULATE TRAJECTORIES
            train_state, env_state, batched_last_obs, rng = runner_state

            # SELECT BATCHED ACTIONS
            rng, _rng = jax.random.split(rng)
            pi, batched_value = network.apply(train_state.params, batched_last_obs)
            batched_action = pi.sample(seed=_rng)
            batched_log_prob = pi.log_prob(batched_action)

            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, num_envs)
            returned_obsv, env_state, batched_reward, done, batch_info = env.step(
                rng_step, env_state, batched_action, env_params
            )
            batched_transition = Transition(
                batched_action,
                batched_value,
                batched_reward,
                batched_log_prob,
                batched_last_obs,
            )

            # We pass the old observations here, it doesn't change in the single-step environment
            runner_state = (train_state, env_state, batched_last_obs, rng)

            batched_advantage = batched_reward - batched_value

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["VALUE_CLIP_EPS"], config["VALUE_CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = num_envs
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[1:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (
                train_state,
                batched_transition,
                batched_advantage,
                batched_reward,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = batch_info
            global_updatestep = metric["timestep"][0]
            rng = update_state[-1]
            if config.get("DEBUG"):

                def return_readout_stats(global_updatestep, info):
                    jax.debug.print("global update: {update}", update=global_updatestep)
                    jax.debug.print(
                        "reward: {reward}",
                        reward=jnp.round(jnp.mean(info["reward"]), 3),
                    )
                    jax.debug.print(
                        "pF reward: {pF}", pF=jnp.round(jnp.mean(info["pF reward"]), 3)
                    )
                    jax.debug.print(
                        "time reward: {time}",
                        time=jnp.round(jnp.mean(info["time reward"]), 4),
                    )
                    jax.debug.print(
                        "smoothness reward: {smoothness}",
                        smoothness=jnp.round(jnp.mean(info["smoothness reward"]), 6),
                    )
                    jax.debug.print(
                        "amp reward: {p}",
                        p=jnp.round(jnp.mean(info["amp reward"]), 3),
                    )
                    jax.debug.print(
                        "max pF: {pF}", pF=jnp.round(jnp.mean(info["max pF"]), 3)
                    )
                    jax.debug.print(
                        "max photon: {photon}",
                        photon=jnp.round(jnp.mean(info["max photon"]), 3),
                    )
                    jax.debug.print(
                        "photon time: {time}",
                        time=jnp.round(jnp.mean(info["photon time"]), 4),
                    )
                    jax.debug.print(
                        "real photon reset time: {time}",
                        time=jnp.round(jnp.mean(info["real photon reset time"]), 4),
                    )
                    jax.debug.print(
                        "smoothness: {smoothness}",
                        smoothness=jnp.round(jnp.mean(info["smoothness"]), 6),
                    )
                    jax.debug.print(
                        "bandwidth: {bandwidth}",
                        bandwidth=jnp.round(jnp.mean(info["bandwidth"]), 3),
                    )
                    jax.debug.print(
                        "pulse reset val: {p}",
                        p=jnp.round(jnp.mean(info["pulse reset val"]), 3),
                    )

                def pass_stats(global_updatestep, info):
                    pass

                def return_action(global_updatestep, info):
                    jax.debug.print(
                        "action of max={action_of_max}",
                        action_of_max=info["action of max"],
                    )

                jax.lax.cond(
                    global_updatestep % config["PRINT_RATE"] == 0,
                    return_readout_stats,
                    pass_stats,
                    global_updatestep,
                    metric,
                )
                if config.get("DEBUG_ACTION"):
                    jax.lax.cond(
                        global_updatestep % config["ACTION_PRINT_RATE"] == 0,
                        return_action,
                        pass_stats,
                        global_updatestep,
                        metric,
                    )

            runner_state = (train_state, env_state, batched_last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    batchsize = 64
    num_envs = 8
    num_updates = 2500
    config = {
        "LR": 3e-3,
        "NUM_ENVS": num_envs * batchsize,
        "NUM_STEPS": 1,
        "NUM_UPDATES": num_updates,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": int(batchsize * num_envs / 64),
        "CLIP_EPS": 0.2,
        "VALUE_CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "relu6",
        "LAYER_SIZE": 256,
        "ENV_NAME": "single_langevin_env",
        "ANNEAL_LR": False,
        "DEBUG": True,
        "DEBUG_ACTION": False,
        "PRINT_RATE": 100,
        "ACTION_PRINT_RATE": 100,
    }

    # Cairo Params
    tau_0 = 0.398
    kappa = 25.0
    chi = 0.65
    kerr = 0.002
    gamma_I = 1 / 500
    gamma = gamma_I
    time_coeff = 10.0
    snr_coeff = 10.0
    smoothness_coeff = 10.0
    rough_max_photons = 30
    actual_max_photons = rough_max_photons * (1 - jnp.exp(-0.5 * kappa * tau_0)) ** 2
    print(actual_max_photons)
    rough_max_amp_scaled = 1 / 0.43
    ideal_photon = 0.05
    num_t1 = 8.0
    scaling_factor = 7.5
    photon_gamma = 1 / 2000
    init_fid = 0.999

    rng = jax.random.PRNGKey(30)
    rng, _rng = jax.random.split(rng)

    single_train = jit(PPO_make_train(config), static_argnums=(-2, -1))

    print(f"Starting a Run of {num_updates} Updates")
    start = time.time()
    single_result = single_train(
        _rng,
        kappa,
        chi,
        kerr,
        gamma,
        time_coeff,
        snr_coeff,
        smoothness_coeff,
        rough_max_photons,
        actual_max_photons,
        rough_max_amp_scaled,
        ideal_photon,
        scaling_factor,
        gamma_I,
        num_t1,
        photon_gamma,
        init_fid,
        # batchsize,
        num_envs,
    )
    print(f"time taken: {time.time() - start}")
