import tensorrt as trt
import torch

weight = torch.load("./models/mnist_state.pt")


def create_network_from_onnx(network: trt.INetworkDefinition, logger: trt.ILogger):
    parser = trt.OnnxParser(network, logger)
    parser.parse_from_file("./models/mnist.onnx")


def create_network(network: trt.INetworkDefinition):
    network.add_input("input", trt.float32, (1, 28, 28))
    conv1 = network.add_convolution(
        network.get_input(0),
        20,
        (5, 5),
        weight["conv1.weight"].cpu().numpy(),
        weight["conv1.bias"].cpu().numpy()
    )
    conv1.stride = (1, 1)
    conv1.name = "conv1"
    relu1 = network.add_activation(
        conv1.get_output(0),
        trt.ActivationType.RELU
    )
    relu1.name = "relu1"
    pool1 = network.add_pooling(
        relu1.get_output(0),
        trt.PoolingType.MAX,
        (2, 2)
    )
    pool1.stride = (2, 2)
    pool1.name = "pool1"
    conv2 = network.add_convolution(
        pool1.get_output(0),
        50,
        (5, 5),
        weight["conv2.weight"].cpu().numpy(),
        weight["conv2.bias"].cpu().numpy()
    )
    conv2.stride = (1, 1)
    conv2.name = "conv2"
    relu2 = network.add_activation(
        conv2.get_output(0),
        trt.ActivationType.RELU
    )
    relu2.name = "relu2"
    pool2 = network.add_pooling(
        relu2.get_output(0),
        trt.PoolingType.MAX,
        (2, 2)
    )
    pool2.stride = (2, 2)
    pool2.name = "pool2"
    full1 = network.add_fully_connected(
        pool2.get_output(0),
        500,
        weight["full1.weight"].cpu().numpy(),
        weight["full1.bias"].cpu().numpy()
    )
    full1.name = "full1"
    relu3 = network.add_activation(
        full1.get_output(0),
        trt.ActivationType.RELU
    )
    relu3.name = "relu3"
    full2 = network.add_fully_connected(
        relu3.get_output(0),
        10,
        weight["full2.weight"].cpu().numpy(),
        weight["full2.bias"].cpu().numpy()
    )
    full2.name = "full2"
    full2.get_output(0).name = "output"

    network.mark_output(full2.get_output(0))
