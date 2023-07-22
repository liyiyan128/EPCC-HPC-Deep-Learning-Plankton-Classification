import torch

if torch.cuda.is_available():
    print("Cuda is available")
else:
    print("Cuda cannot be found")
