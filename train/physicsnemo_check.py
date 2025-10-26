import torch
from physicsnemo.models.mlp.fully_connected import FullyConnected
model = FullyConnected(in_features=32, out_features=64)
input = torch.randn(128, 32)
output = model(input)
print(output.shape)
