import cv2
import pycuda
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np

from PIL import Image


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class DetectNet(object):
    def __init__(self, trt_file_path, input_size=(960, 544), num_class=3,
                 batch_size=1, box_norm=35.0, stride=16):     
        self.trt_path = trt_file_path
        self.input_size = input_size
        self.batch_size = batch_size
        self.box_norm = box_norm
        self.stride = stride
        self.num_class = num_class

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt_runtime = trt.Runtime(TRT_LOGGER)
        self.trt_engine = self._load_engine(trt_runtime, self.trt_path)
        self.inputs, self.outputs, self.bindings, self.stream = \
            self._allocate_buffers()

        self.context = self.trt_engine.create_execution_context()


    def _load_engine(self, trt_runtime, engine_path):
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine


    def _allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        binding_to_type = {
            "input_1": np.float32,
            "output_bbox/BiasAdd": np.float32,
            "output_cov/Sigmoid": np.float32}

        for binding in self.trt_engine:
            size = trt.volume(self.trt_engine.get_binding_shape(binding)) \
                * self.batch_size
            dtype = binding_to_type[str(binding)]
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.trt_engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream


    def _do_inference(self, context, bindings, inputs,
                      outputs, stream):
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) \
            for inp in inputs]
        context.execute_async(
            batch_size=self.batch_size, bindings=bindings,
            stream_handle=stream.handle)

        [cuda.memcpy_dtoh_async(out.host, out.device, stream) \
            for out in outputs]

        stream.synchronize()

        return [out.host for out in outputs]


    def _process_image(self, arr, w, h):
        image = Image.fromarray(np.uint8(arr))
        image_resized = image.resize(size=(w, h), resample=Image.BILINEAR)
        img_np = np.array(image_resized)
        img_np = img_np.transpose((2, 0, 1))
        img_np = (1.0 / 255.0) * img_np
        img_np = img_np.ravel()

        return img_np

    def _check_bbox(self, bboxes, scores, min_size):
        w_min_box = min_size[0]
        h_min_box = min_size[1]
        new_bboxes, new_scores = [], []
        
        box_id = 0
        for bbox in bboxes:
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(self.input_size[0], bbox[2])
            bbox[3] = min(self.input_size[1], bbox[3])
            w_box = bbox[2]
            h_box = bbox[3]
            if w_box > w_min_box or h_box > h_min_box :
                new_bboxes.append(bbox)
                new_scores.append(scores[box_id])
        
        return new_bboxes, new_scores
    

    def _add_margin_scale(self, bbox, w_img, h_img):
        x_scale = w_img / self.input_size[0]
        y_scale = h_img / self.input_size[1]

        x_min = int(bbox[0])
        y_min = int(bbox[1])
        x_max = int(bbox[2]+bbox[0])
        y_max = int(bbox[3]+bbox[1])

        (Left, Top, Right, Bottom) = (x_min, y_min, x_max, y_max)
        x_min = int(np.round(Left * x_scale))
        y_min = int(np.round(Top * y_scale))
        x_max = int(np.round(Right * x_scale))
        y_max = int(np.round(Bottom * y_scale))
        rescale_bbox = [x_min, y_min, x_max, y_max]

        return rescale_bbox

    @staticmethod
    def _scale_box(box, image_width, image_height, scale_x=1.0, scale_y=1.0):
        width = box[2] - box[0]
        height = box[3] - box[1]

        width_scale = ((scale_x - 1) / 2) * width
        height_scale = ((scale_y - 1) / 2) * height

        scaled_box = [int(max(box[0] - width_scale, 0)),
                      int(max(box[1] - height_scale, 0)),
                      int(min(box[2] + width_scale, image_width)),
                      int(min(box[3] + height_scale, image_height))]

        return scaled_box


    def predict(self, image, confidence=0.1, min_size=(30, 30),
                nms_threshold=0.5, scale_x=1.0, scale_y=1.0):
        self.list_output = []
        w, h = self.input_size[0], self.input_size[1]
        h_img, w_img, _ = image.shape
        img = self._process_image(image, w, h)
        np.copyto(self.inputs[0].host, img.ravel())

        [detection_out, keepCount_out] = self._do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream)
        bboxes, class_ids, scores = self._postprocess(
            [detection_out, keepCount_out],
            confidence, list(range(self.num_class)))

        new_bboxes, new_scores = self._check_bbox(bboxes, scores, min_size)
        box_indexes = cv2.dnn.NMSBoxes(new_bboxes, new_scores,
                                       confidence, nms_threshold)

        for idx in box_indexes:
            idx = int(idx)
            rescale_box = self._add_margin_scale(new_bboxes[idx], w_img, h_img)
            scaled_box = self._scale_box(rescale_box, w_img, h_img,
                                         scale_x, scale_y)
            score = np.float32(scores[idx])
            res = {"calss_id": class_ids[idx],
                   "confidence": round(np.float32(score).item(), 2),
                   "bounding_box": scaled_box}
            self.list_output.append(res)
        
        return self.list_output


    def _applyBoxNorm(self, o1, o2, o3, o4, x, y,
                      grid_centers_w, grid_centers_h):
        o1 = (o1 - grid_centers_w[x]) * -self.box_norm
        o2 = (o2 - grid_centers_h[y]) * -self.box_norm
        o3 = (o3 + grid_centers_w[x]) * self.box_norm
        o4 = (o4 + grid_centers_h[y]) * self.box_norm

        return o1, o2, o3, o4


    def _compute_grids(self):
        grid_h = int(self.input_size[1] / self.stride)
        grid_w = int(self.input_size[0] / self.stride)
        
        grid_size = grid_h * grid_w

        grid_centers_w = []
        grid_centers_h = []

        for i in range(grid_h):
            value = (i * self.stride + 0.5) /self.box_norm
            grid_centers_h.append(value)

        for i in range(grid_w):
            value = (i * self.stride + 0.5) / self.box_norm
            grid_centers_w.append(value)

        return grid_w, grid_h, grid_size, grid_centers_w, grid_centers_h


    def _postprocess(self, outputs, min_confidence,
                     analysis_classes, wh_format=True):
        bbs = []
        class_ids = []
        scores = []
        grid_w , grid_h, grid_size, grid_centers_w, grid_centers_h = \
            self._compute_grids()
        for c in analysis_classes:
            x1_idx = c * 4 * grid_size
            y1_idx = x1_idx + grid_size
            x2_idx = y1_idx + grid_size
            y2_idx = x2_idx + grid_size

            boxes = outputs[0]
            for h in range(grid_h):
                for w in range(grid_w):
                    i = w + h * grid_w
                    score = outputs[1][c * grid_size + i]
                    if score >= min_confidence:
                        o1 = boxes[x1_idx + w + h * grid_w]
                        o2 = boxes[y1_idx + w + h * grid_w]
                        o3 = boxes[x2_idx + w + h * grid_w]
                        o4 = boxes[y2_idx + w + h * grid_w]

                        o1, o2, o3, o4 = self._applyBoxNorm(
                            o1, o2, o3, o4, w, h,
                            grid_centers_w, grid_centers_h)

                        xmin = int(o1)
                        ymin = int(o2)
                        xmax = int(o3)
                        ymax = int(o4)
                        if wh_format:
                            bbs.append([xmin, ymin, xmax - xmin, ymax - ymin])
                        else:
                            bbs.append([xmin, ymin, xmax, ymax])
                        class_ids.append(c)
                        scores.append(float(score))

        return bbs, class_ids, scores
