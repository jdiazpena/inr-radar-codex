from models import MLPINR
import torch

model = MLPINR(
    in_features=2,
    out_features=1,
    hidden_features=256,
    hidden_layers=3,
    activation="relu",
)

x = torch.randn(10, 2)
y = model(x)

print("input shape:", x.shape)
print("output shape:", y.shape)

# from datasets import ImageINRDataset

# ds = ImageINRDataset(sidelength=256)
# sample = ds[0]

# print("coords shape:", sample["coords"].shape)
# print("values shape:", sample["values"].shape)
# print("coords min/max:", sample["coords"].min().item(), sample["coords"].max().item())
# print("values min/max:", sample["values"].min().item(), sample["values"].max().item())

# print(sample["coords"][:5])
# print(sample["values"][:5])