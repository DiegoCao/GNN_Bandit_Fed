import numpy as np
from .bandit import ContextualBandit
from .neuralucb import NeuralUCB
import itertools
import matplotlib.pyplot as plt


# Bandit settings
class Bandit:
    def __init__(self, T, n_arms, n_features, features, use_cuda):
        self.bandit = ContextualBandit(T, n_arms, n_features, noise_std=0.1, seed=42, features=features)
        self.model = NeuralUCB(self.bandit,
                               hidden_size=16,
                               reg_factor=1.0,
                               delta=0.1,
                               confidence_scaling_factor=1,
                               training_window=100,
                               p=0.2,
                               learning_rate=0.01,
                               epochs=100,
                               train_every=1,
                               use_cuda=use_cuda
                               )

    def get_arm(self, iter):
        return self.model.per_run_1(iter)

    def set_reward(self, iter, r):
        self.model.per_run_2(iter, r)
