import torch
import numpy as np

"""
t0 = torch.empty(1,2,3)
print(t0)
print(t0.size())  # torch.Size([1, 2, 3])

print(t0.size(dim=2))  # 3
print(type(t0.size(dim=2)))  # int

print(np.pi)
print(type(np.pi))
"""

a = torch.randn(4, 2)
b = torch.randn(4, 2)
print(a)
print(b)

idx = torch.argmin(
    torch.cat(
        (torch.sum(torch.pow(a, 2), dim=1, keepdim=True), torch.sum(torch.pow(b, 2), dim=1, keepdim=True)),
        dim=1),
    dim=1)
print(idx)

idx = torch.unsqueeze(idx, dim=1)
print(idx)

m = torch.where(idx == 0, a, b)

print(m)
