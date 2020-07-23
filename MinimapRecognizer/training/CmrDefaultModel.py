import os
import cv2
from MinimapRecognizer.training.losses import MulticlassDiceLoss
from MinimapRecognizer.training.model import MRNetwork
from tensorflow import keras

class CmrDefaultModel:
  def __init__(self):
    input_shape = self.input_shape
    
    self.network = MRNetwork(input_shape)
    return
  
  @property
  def input_shape(self):
    return (256, 256, 3)
  
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
    return CmrdTrainingParameters()
  
  def predict(self, samples):
    # must return (samples N, h, w, 2)
    res = self.network.predict(samples)
    return res[:, :, :, 1:] # ignore first class
  
  def preprocessing(self, bgrImage):
    return cv2.cvtColor(bgrImage, cv2.COLOR_BGR2Lab) / 255. # normalize
  
####################################
class CmrdTrainingParameters:
  def __init__(self):
    return
  
  def loss(self):
    dice = MulticlassDiceLoss(weights=[1., 1., 1.])
    def calc(y_true, y_pred):
      dloss = dice(
        y_true,
        keras.backend.pow(y_pred, 2) # push "down" predictions
      )
      return dloss
    return calc

  def DataGenerator(self, generator):
    return generator
