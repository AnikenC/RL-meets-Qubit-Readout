# Enhanced qubit readout via reinforcement learning - Codebase

This repository provides code for the paper, [Enhanced qubit readout via reinforcement learning](https://doi.org/10.1103/PhysRevApplied.23.054057).

## Installation

All requirements for the package can be installed from the requirements.txt file. The requirements don't explicitly account for GPU support, however by updating the jax installation to use cuda all of the code will be compatible.

## Usage

The MLControl/example_kyoto.ipynb notebook can be used to complete a training run and analyse the results according to the Langevin-based readout model used in the paper. Note that the exact waveform differs from the original paper to the seed used.

## WIP

Further notebooks will be added, including reproducing Figures 2 and 3 for the Brisbane and Kyoto Devices respectively.

Additionally, example code will be provided for comparing the RL-discovered waveforms with the CLEAR Pulse from the 2016 [Rapid Driven Reset of a Qubit Readout Resonator](https://doi.org/10.1103/PhysRevApplied.5.011001) paper.
