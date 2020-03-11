from keras.models import model_from_json
import tensorflow as tf
import rospkg
import numpy as np
class SignClassifier:
    def __init__(self):
        self.graph = tf.get_default_graph()
        rospack = rospkg.RosPack()
        cur_dir = rospack.get_path('team503')
        json_file = open(cur_dir + '/scripts/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(cur_dir + "/scripts/model.h5")
        
    def init_classifier(self):
        with self.graph.as_default():
            self.model.predict(np.ones((1,16,16,3)))
        print("Loaded sign classifier model from disk")