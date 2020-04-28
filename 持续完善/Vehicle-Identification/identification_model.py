from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from keras.models import load_model


class New_Session_Model(object):
    def __init__(self, model_path):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session(graph=self.graph)
            with self.session.as_default():
                self.model = load_model(model_path)

    def predict(self, X):
        with self.graph.as_default():
            with self.session.as_default():
                return self.model.predict(X)
