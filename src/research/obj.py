# Import packages
import os
import sys
#from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util
import cv2
import numpy as np
import tensorflow as tf

from src.research.object_detection.utils import label_map_util
from src.research.object_detection.utils import visualization_utils as vis_util
from src.utils.utils import encodeImageIntoBase64

class MultiClassObj:

    def __init__(self, imagePath, modelPath):
        """cloth classification 

        Args:
            imagePath (str): path of the input image
            modelPath (str): model file path
        """
        
        sys.path.append("..")
        self.MODEL_NAME = modelPath
        self.IMAGE_NAME = imagePath
        
        CWD_PATH = os.getcwd()
        self.PATH_TO_CKPT = os.path.join(CWD_PATH, self.MODEL_NAME, 'frozen_inference_graph.pb')
        # Path to label map file
        self.PATH_TO_LABELS = os.path.join(CWD_PATH, 'src/research/data', 'labelmap.pbtxt')
       
        self.PATH_TO_IMAGE = os.path.join(CWD_PATH, self.IMAGE_NAME)
        print(self.PATH_TO_IMAGE)
        
        self.NUM_CLASSES = 11

        
        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
                                                                         max_num_classes=self.NUM_CLASSES,
                                                                         use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        self.class_names_mapping = {
            1: "ear_ring", 2: "GirlsTopWear", 3: "glass", 4: "hat", 5: "Jacket", 6: "MensShorts", 7: "MensTopWear",
            8: "Pant", 9: "shoes", 10: "tie", 11: "watch"
        }
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

       
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

       
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def getPrediction(self):
       
        sess = tf.Session(graph=self.detection_graph)
        image = cv2.imread(self.PATH_TO_IMAGE)
        image_expanded = np.expand_dims(image, axis=0)
        
        (boxes, scores, classes, num) = sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})

        result = scores.flatten()
        res = []
        for idx in range(0, len(result)):
            if result[idx] > .40:
                res.append(idx)

        top_classes = classes.flatten()
        res_list = [top_classes[i] for i in res]

        class_final_names = [self.class_names_mapping[x] for x in res_list]
        top_scores = [e for l2 in scores for e in l2 if e > 0.30]
        new_scores = scores.flatten()
        new_boxes = boxes.reshape(300, 4)

        # get all boxes from an array
        max_boxes_to_draw = new_boxes.shape[0]
       
        min_score_thresh = .30

        listOfOutput = []
        for (name, score, i) in zip(class_final_names, top_scores, range(min(max_boxes_to_draw, new_boxes.shape[0]))):
            valDict = {}
            valDict["className"] = name
            valDict["confidence"] = str(score)
            if new_scores is None or new_scores[i] > min_score_thresh:
                val = list(new_boxes[i])
                valDict["yMin"] = str(val[0])
                valDict["xMin"] = str(val[1])
                valDict["yMax"] = str(val[2])
                valDict["xMax"] = str(val[3])
                listOfOutput.append(valDict)

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
        output_filename = 'output_image.jpg'
        cv2.imwrite(output_filename, image)
        opencodedbase64 = encodeImageIntoBase64("output_image.jpg")
 
  
        listOfOutput.append({"image" : opencodedbase64.decode('utf-8')})
        return listOfOutput

