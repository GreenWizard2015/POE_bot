from .model import StackedMRModel
import os
import cv2
from .losses import MulticlassDiceLoss
from .CmrModelA import CmrModelA
from .CmrModelB import CmrModelB

class CmrDefaultModel:
  def __init__(self):
    input_shape = self.input_shape
    
    self.subnetA = CmrModelA()
    self.subnetB = CmrModelB()
    
    self.network, _, _ = StackedMRModel(
      input_shape,
      [self.subnetA.network, self.subnetB.network]
    )
    return
  
  @property
  def input_shape(self):
    return (64, 64, 3)
  
  @property
  def weights_file(self):
    return os.path.join(
      os.path.dirname(os.path.dirname(__file__)),
      'weights',
      'stacked.h5'
    )

  def load(self, only_fully_trained, reset=False):
    if os.path.exists(self.weights_file) and not reset:
      self.network.load_weights(self.weights_file)
    else:
      if only_fully_trained:
        raise Exception('There is no fully trained model Stacked.')
      
      self.subnetA.load(only_fully_trained=True)
      self.subnetB.load(only_fully_trained=True)

    self.subnetA.lock()
    self.subnetB.lock()
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
    return MulticlassDiceLoss(weights=[
      0., # background totally ignored
      .4, # we have much more walls, so they are less important
      .6  # prioritize undiscovered areas  
    ])

  def DataGenerator(self, generator):
    return generator
