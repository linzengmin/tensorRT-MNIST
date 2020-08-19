import os
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

from utils.model import create_network


def create_engine(runtime, data_type, logger):
    engine_type = {
        trt.DataType.FLOAT: 'FLOAT',
        trt.DataType.HALF: 'HALF',
        trt.DataType.INT32: 'INT32',
        trt.DataType.INT8: 'INT8'
    }
    engine_path = './models/engine_{}.bin'.format(engine_type[data_type])
    if os.path.exists(engine_path):
        print("engine found, restoring...")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
    else:
        with trt.Builder(logger) as builder, builder.create_network() as network:
            builder.max_workspace_size = 1 << 30
            if data_type == trt.DataType.HALF:
                builder.fp16_mode = True
            builder.max_batch_size = 1

            print("Creating network...")
            create_network(network)

            print("Building engine...")

            engine = builder.build_cuda_engine(network)

            buf = engine.serialize()
            with open(engine_path, 'wb') as f:
                f.write(buf)

    return engine


def allocate_buffers(engine):
    """Allocates host and device buffer for TRT engine inference.

    This function is similair to the one in ../../common.py, but
    converts network outputs (which are np.float32) appropriately
    before writing them to Python buffer. This is needed, since
    uff plugins doesn't support output type description, and
    in our particular case, we use NMS plugin as network output.

    Args:
        engine (trt.ICudaEngine): uff engine

    Returns:
        inputs [HostDeviceMem]: engine input memory
        outputs [HostDeviceMem]: engine output memory
        bindings [int]: buffer to device bindings
        stream (cuda.Stream): cuda stream for engine inference synchronization
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    # Current NMS implementation in TRT only supports DataType.FLOAT but
    # it may change in the future, which could brake this sample here
    # when using lower precision [e.g. NMS output would not be np.float32
    # anymore, even though this is assumed in binding_to_type]

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
