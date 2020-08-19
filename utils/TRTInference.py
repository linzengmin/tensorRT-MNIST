import numpy as np

from .engine import create_engine, allocate_buffers, do_inference
import tensorrt as trt
import cv2


def process_image(img):
    model_input_width = 28
    model_input_height = 28
    img_np = cv2.resize(img, (model_input_width, model_input_height))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    # img_np = (2.0 / 255.0) * img_np - 1.0
    # cv2.normalize(img_np, img_np, -1.0, 1.0, cv2.NORM_MINMAX)
    # img_np = np.expand_dims(img_np, axis=0)
    # img_np = img_np.ravel()
    return img_np


class TRTInference(object):
    engine_type = {
        trt.DataType.HALF: "HALF",
        trt.DataType.FLOAT: "FLOAT",
        trt.DataType.INT32: "INT32",
        trt.DataType.INT8: "INT8"
    }

    def __init__(self, data_type=trt.DataType.FLOAT, debug=False):
        self.data_type = data_type
        self.debug = debug
        if debug:
            self.logger = trt.Logger(trt.Logger.VERBOSE)
        else:
            self.logger = trt.Logger()
        # trt.init_libnvinfer_plugins(self.logger, '')
        self.runtime = trt.Runtime(self.logger)

        self.engine = create_engine(self.runtime, self.data_type, self.logger)
        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        # Execution context is needed for inference
        self.context = self.engine.create_execution_context()

    def inference(self, img):
        image = process_image(img)
        np.copyto(self.inputs[0].host, image.ravel())
        [output] = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream)

        pred = np.argmax(output)

        return pred
