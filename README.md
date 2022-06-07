# Understanding-Adversarial-Examples-Through-DNNs-Classification-Boundary-and-Uncertainty-Regions

The paper is submitted to SafetyAI 2022.


## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)

## Background

In this paper we first study DNN’s classification boundary. Through experiments, we show the problem of adversarial examples is not as simple as linear vs. non-linear. It is a far more complex structural problem.

## Install

We use pytorh and foolbox for all the experiments in the paper. Find more about installaion here: [pytorch](https://pytorch.org/get-started/locally/), [foolbox](https://foolbox.jonasrauber.de/guide/getting-started.html)

## Usage

The experimens in the paper can be devided into several steps and users can run the code sequentially.

- Train 10 CNN models, M1, M2, ..., M10, which have same model structure but different seeds.
- Attack M1 and generate n adversarial images for each attack algorithms listed in the paper according to different hyper-parameters of the attack algorithm.
- Explore the uncertainty region
- Generate new adversarial images based on uncertainty regions and test on M2 to M10.
