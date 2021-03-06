# Some standard imports
import io
import numpy as np
from torch import nn
import torch.onnx
import torchvision
import onnx
import model
# Standard ImageNet input - 3 channels, 224x224,
# values don't matter as we care about network structure.
# But they can also be real inputs.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.load('./models/mnist_model_10.pt').to(device)
model = torch.load('./models/mnist_model_10.pt',map_location=lambda storage, loc: storage)
#model=model.Model().to(device)
#model.load_state_dict(torch.load('./models/mnist_model_10.pt'))
model.eval()
# Obtain your model, it can be also constructed in your script explicitly

#model = torchvision.models.alexnet(pretrained=True)
# Invoke export
dummy_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(model, dummy_input, "mnist_model_10.onnx",verbose=True)
