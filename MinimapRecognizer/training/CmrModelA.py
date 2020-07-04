from .model import MRModel
import os
import cv2
from .losses import MulticlassDiceLoss
import numpy
from training.CdgRemoveLayer import CdgRemoveLayer

class CmrModelA:
  def __init__(self):
    input_shape = self.input_shape
    self.network = MRModel(input_shape, 2)
    return
  
  @property
  def input_shape(self):
    return (64, 64, 3)
  
  @property
  def weights_file(self):
    return os.path.join(
      os.path.dirname(os.path.dirname(__file__)),
      'weights',
      'modelA.h5'
    )

  def load(self, only_fully_trained, reset=False):
    if os.path.exists(self.weights_file) and not reset:
      self.network.load_weights(self.weights_file)
    else:
      if only_fully_trained:
        raise Exception('There is no fully trained model A.')
    return True

  def lock(self):
    for layer in self.network.layers:
      layer.trainable = False
    return

  def trainingParams(self):
    return CmrdTrainingParameters()
  
  def predict(self, samples):
    # must return (samples N, h, w, 2)
    res = self.network.predict(samples)

    res[:, :, :, 0] = res[:, :, :, 1] # walls mask must be first
    res[:, :, :, 1] = numpy.zeros(res[:, :, :, 1].shape) # hide "unknown" class
    return res

  def preprocessing(self, bgrImage):
    return cv2.cvtColor(bgrImage, cv2.COLOR_BGR2Lab) / 255. # normalize
  
################
class CmrdTrainingParameters:
  def __init__(self):
    return
  
  def loss(self):
    return MulticlassDiceLoss(weights=[0., 1.]) # ignore first class
  
  def DataGenerator(self, generator):
    return CdgRemoveLayer(generator, layer=2)
