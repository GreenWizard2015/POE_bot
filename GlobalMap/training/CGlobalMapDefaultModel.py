import os
from GlobalMap.training.model import GMNetwork
from MinimapRecognizer.training.losses import MulticlassDiceLoss
import tensorflow.keras.backend as K

class CGlobalMapDefaultModel:
  def __init__(self):
    self.network = GMNetwork((256, 256, 1))
    return
  
  @property
  def weights_file(self):
    return os.path.join(
      os.path.dirname(os.path.dirname(__file__)),
      'weights',
      'main.h5'
    )

  def load(self, only_fully_trained, reset=False):
    if os.path.exists(self.weights_file) and not reset:
      self.network.load_weights(self.weights_file)
    else:
      if only_fully_trained:
        raise Exception('There is no fully trained model Stacked.')

    return True

  def trainingParams(self):
    return CTrainingParameters()
  
  def predict(self, samples):
    # must return (samples N, h, w, 2)
#     res = self.network.predict(samples)
#     return res[:, :, :, 1:] # ignore first class
    return
  
####################################
class CTrainingParameters:
  def __init__(self):
    return
  
  def loss(self):
    def calc(y_true, y_pred):
      return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
    return calc

  def DataGenerator(self, generator):
    return generator