#
# refactored from https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/__init__.py
# focused on yolov3 applications only
#
# Van Hoey S.
#

import os
import cv2
import numpy as np
from PIL import Image
from keras import backend as K
from keras.layers import Input

from models import yolo_main
from utils import yolo_eval, draw_box, draw_caption


class YoloObjectDetection:

    speed_options = ["normal", "fast", "faster", "fastest", "flash"]

    numbers_to_names = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
        7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
        13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
        27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
        34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
        39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
        46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
        52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
        59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
        66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
        72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
        78: 'hair drier', 79: 'toothbrush'}

    _yolo_iou = 0.45
    _yolo_score = 0.1
    _yolo_anchors = np.array([[10., 13.], [16., 30.], [33., 23.],
                              [30., 61.], [62., 45.], [59., 119.],
                              [116., 90.], [156., 198.], [373., 326.]])

    def __init__(self, path_model, detection_speed="normal"):
        if os.path.isfile(path_model) :
            self._model_path = path_model

        # placeholders for config
        self._yolo_boxes, self._yolo_scores, self._yolo_classes = None, None, None
        self._yolo_input_image_shape = K.placeholder(shape=(2,))

        # load the model
        self._yolo_model_image_size = (416, 416)
        self.detection_speed = detection_speed

        # Keras session
        self.sess = K.get_session()

    @property
    def model_path(self):
        """Location of the yolo `.h5` model
        """
        return self._model_path

    @property
    def detection_speed(self):
        """Making detection faster by decreasing image resolution
        """
        return self._detection_speed

    @detection_speed.setter
    def detection_speed(self, speed):
        if speed not in self.speed_options:
            raise Exception("Invalid speed parameter used.")
        else:
            self._detection_speed = speed

            if (self._detection_speed == "normal"):
                self._yolo_model_image_size = (416, 416)
            elif (self._detection_speed == "fast"):
                self._yolo_model_image_size = (320, 320)
            elif (self._detection_speed == "faster"):
                self._yolo_model_image_size = (208, 208)
            elif (self._detection_speed == "fastest"):
                self._yolo_model_image_size = (128, 128)
            elif (self._detection_speed == "flash"):
                self._yolo_model_image_size = (96, 96)
            self._model = self.prepare_model(self._detection_speed)

    def prepare_model(self, detection_speed="normal"):
        """Load the model with a given detection speed
        """

        model = yolo_main(Input(shape=(None, None, 3)),
                          len(self._yolo_anchors) // 3,
                          len(self.numbers_to_names))
        model.load_weights(self.model_path)

        (self._yolo_boxes, self._yolo_scores, self._yolo_classes) = \
            yolo_eval(model.output, self._yolo_anchors, len(self.numbers_to_names),
            self._yolo_input_image_shape, score_threshold = self._yolo_score,
            iou_threshold = self._yolo_iou)
        return model

    @staticmethod
    def read_image_array(image_array):
        """Preprocess array as int8 array
        """
        image = np.asarray(Image.fromarray(np.uint8(image_array)))
        return image[:, :, ::-1].copy()

    @staticmethod
    def letterbox_image(image, size):
        """Letterbox the image
        """
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))

        return new_image

    def detect_objects(self, input_image,
                       minimum_percentage_probability = 50,
                       custom_objects = ["person", "car", "bicycle", "motorcycle"]):
        """Detect objects on a given input image

        Parameters
        ----------
        input_image : np.array
            Threedimensional array with the RGB channels of the image
        minimum_percentage_probability : int [0-100]
            Required probablity to include detected objects in the result
        custom_objects : list of char
            List containing the object classes to identify, e.g.
            'person', 'car',...

        Returns
        -------
        detected_objects : list of dicts
            For each detected object, a dict with the class name, detection
            probability and the box points of the containing box
        """
        detected_objects = []

        # prepare the image
        image = Image.fromarray(np.uint8(input_image))
        input_image = self.read_image_array(input_image)

        new_image_size = (self._yolo_model_image_size[0] - (self._yolo_model_image_size[0] % 32),
                            self._yolo_model_image_size[1] - (self._yolo_model_image_size[1] % 32))
        boxed_image = self.letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype="float32")

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        model = self._model

        out_boxes, out_scores, out_classes = self.sess.run(
            [self._yolo_boxes, self._yolo_scores, self._yolo_classes],
            feed_dict={
                model.input: image_data,
                self._yolo_input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        min_probability = minimum_percentage_probability / 100

        counting = 0
        for a, b in reversed(list(enumerate(out_classes))):
            predicted_class = self.numbers_to_names[b]
            box = out_boxes[a]
            score = out_scores[a]
            print(predicted_class, "with probability:", score)

            if score < min_probability:
                continue
            if predicted_class not in custom_objects:
                continue

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            detection_details = (left, top, right, bottom)
            each_object_details = {}
            each_object_details["name"] = predicted_class
            each_object_details["percentage_probability"] = score * 100
            each_object_details["box_points"] = detection_details

            detected_objects.append(each_object_details)
            counting += 1

        return detected_objects

    @staticmethod
    def plot_objects(image, objects_info,
                     display_meta_info = False):
        """Add objects information to input image

        Parameters
        ----------
        image : np.array
            Image to put the object info on
        objects : list of dicts
            For each detected object, a dict with the class name,
            probability and box_points info. Box points info is a tuple
            (left, top, right, bottom) to create the box around the detected
            objects
        display_meta_info : bool
            If True, add info (class name and percentage) about recognition on
            the image
        """

        annotated_image = image.copy()
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        for object in objects_info:
            detection_box = object["box_points"]

            # create a random color
            color = np.random.rand(3,)

            # add box
            draw_box(annotated_image, detection_box, color=color)

            if display_meta_info:
                label = "{} {:.2f}".format(object["name"],
                                           object["percentage_probability"])
                draw_caption(annotated_image, detection_box, label)

        return annotated_image

    # TODO
    @staticmethod
    def anonymize_image(image, objects_info):
        """Add objects information to input image

        Parameters
        ----------
        image : np.array
            Image to put the object info on
        objects : list of dicts
            For each detected object, a dict with the class name,
            probability and box_points info. Box points info is a tuple
            (left, top, right, bottom) to create the box around the detected
            objects
        display_meta_info : bool
            If True, add info (class name and percentage) about recognition on
            the image
        """

        annotated_image = image.copy()
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        for object in objects_info:
            detection_box = object["box_points"]

            # add ....# TODO
            draw_box(annotated_image, detection_box, color=color)

        return annotated_image
