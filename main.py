from utils.TRTInference import TRTInference
import tensorrt as trt
import time
import cv2

if __name__ == '__main__':
    model = TRTInference(data_type=trt.DataType.HALF)
    for i in range(10):
        img = cv2.imread("./digits/{}.png".format(i))
        start_time = time.time()
        result = model.inference(img)
        end_time = time.time() - start_time
        print("Digit {} result: {}, inference time: {:.2f} ms".format(i, result, end_time * 1000))
