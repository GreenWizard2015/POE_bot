import os
import tensorflow.keras.backend as K
from training.losses import MulticlassDiceLoss
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

def convBlock(prev, sz, filters, strides=1):
  conv_1 = layers.Convolution2D(filters, (sz, sz), padding="same", activation="relu", strides=strides)(prev)
  conv_1 = layers.Dropout(0.1)(conv_1)
  conv_1 = layers.BatchNormalization()(conv_1)
  return conv_1

def downsamplingBlockWithLink(prev, sz, filters):
  link = convBlock(prev, sz, filters)
  res = layers.Convolution2D(filters, (2,2), strides=2, padding="same")(link)
  return link, res
  
def upsamplingBlock(prev, shortcut, sz, filters):
  prev = layers.Convolution2DTranspose(filters, (2, 2), strides=2)(prev)
  concatenated = layers.Concatenate()([prev, shortcut])
   
  return convBlock(concatenated, sz, filters)

def MKNetwork(input_shape):
  inputs = layers.Input(shape=input_shape)
  
  res = convBlock(inputs, sz=3, filters=16, strides=2)
  res = convBlock(res, sz=3, filters=32, strides=2)
  
  res = layers.Convolution2D(1, (1, 1), padding="same", activation="sigmoid", strides=1)(res)
  return keras.Model(inputs=inputs, outputs=res)

class CMapKeypointsModel:
  def __init__(self):
    self.network = MKNetwork((256, 256, 1))
    return
  
  @property
  def weights_file(self):
    return os.path.join(
      os.path.dirname(os.path.dirname(__file__)),
      'weights',
      'keypoints.h5'
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
    return MulticlassDiceLoss([1])

  def DataGenerator(self, generator):
    return generator
