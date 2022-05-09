import numpy as np
import matplotlib.pyplot as plt
import seaborn
import ipdb
from torch import distributions as dis
import torch
import itertools

"""
    generate 2d gaussian around a circle
"""
def sample_ring(batch_size, n_mixture=8, std=0.01, radius=1.0):
    """Gnerate 2D Ring"""
#     std = [std, std]
#     std = torch.tensor(std)
    thetas = np.linspace(0, 2 * np.pi, n_mixture + 1)[:-1]
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
#     cat = dis.categorical.Categorical(logits = torch.zeros(n_mixture))
    cat = dis.categorical.Categorical(logits = torch.zeros(n_mixture))

    mean = torch.transpose(torch.tensor([xs.ravel(), ys.ravel()]), 0, 1)
#     print(mean)
    null = torch.empty(n_mixture, 2)
    std = torch.zeros_like(null) + std
    comps = dis.Independent(dis.normal.Normal(mean, std),1)
    data = dis.MixtureSameFamily(cat, comps)
    return data.sample(sample_shape=torch.tensor([batch_size]))


def sample_grid(batch_size, num_components=25, std=0.05):
    """Generate 2D Grid"""
    cat = dis.categorical.Categorical(logits = torch.zeros(num_components))
    mus = np.array([np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),
                                                                   range(-4, 5, 2))], dtype=np.float32)
    mean = torch.tensor(mus)
    null = torch.empty(num_components, 2)
    std = torch.zeros_like(null) + std
    comps = dis.Independent(dis.normal.Normal(mean, std),1)
    data = dis.MixtureSameFamily(cat, comps)
    return data.sample(sample_shape=torch.tensor([batch_size]))