import cv2

from detectnet_v2 import DetectNet


def set_parameters(model_type):
    if model_type == "detectnet_v2":
        height = 544
        width = 960
        trt_path = ("/workspace/test_tlt/model/"
                    "resnet34_peoplenet_pruned_b1_int8.trt")
        num_class = 2
        batch_size = 1
        box_norm = 35.0
        stride = 16

    return (width, height), trt_path, num_class, batch_size, box_norm, stride


if __name__ == "__main__":
    img_path = "/workspace/python_inference/people.jpg"

    min_confidence = 0.5
    min_size = (30, 30)
    nms_threshold = 0.5
    scale_x = 1.3
    scale_y = 1.2

    model_type = "detectnet_v2"

    model_size, trt_path, num_class, batch_size, box_norm, stride = \
        set_parameters(model_type)


    detectnet = DetectNet(trt_path, model_size, num_class,
                              batch_size, box_norm, stride)

    image = cv2.imread(img_path)[..., ::-1]

    list_outputs = detectnet.predict(image, min_confidence, min_size,
                                         nms_threshold, scale_x, scale_y)

    print('\noutputs : {}'.format(list_outputs))
