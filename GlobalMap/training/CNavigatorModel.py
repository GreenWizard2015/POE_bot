import os
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy
from Utils.losses import MulticlassDiceLoss

def convBlock(prev, sz, filters):
  conv_1 = layers.Convolution2D(filters, (sz, sz), padding="same", activation="relu")(prev)
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

def Network(input_shape):
  res = inputs = layers.Input(shape=input_shape)
  
  convA, res = downsamplingBlockWithLink(res, 3, 4)
  convB, res = downsamplingBlockWithLink(res, 3, 8)
  convC, res = downsamplingBlockWithLink(res, 3, 16)
  convD, res = downsamplingBlockWithLink(res, 3, 24)
  
  res = convBlock(res, 3, 32)
  
  res = upsamplingBlock(res, convD, 3, 24)
  res = upsamplingBlock(res, convC, 3, 16)
  res = upsamplingBlock(res, convB, 3, 8)
  res = upsamplingBlock(res, convA, 3, 4)

  return keras.Model(
    inputs=inputs,
    outputs=layers.Conv2D(2, 1, activation='softmax', padding='same')(res)
  )
  
class CNavigatorModel:
  def __init__(self):
    self.network = Network((128, 128, 3))
    return
  
  @property
  def weights_file(self):
    return os.path.join(
      os.path.dirname(os.path.dirname(__file__)),
      'weights',
      'navigator.h5'
    )

  def load(self, only_fully_trained, reset=False):
    if os.path.exists(self.weights_file) and not reset:
      self.network.load_weights(self.weights_file)
    else:
      if only_fully_trained:
        raise Exception('There is no fully trained model.')

    return True

  def trainingParams(self):
    return CTrainingParameters()
  
  def predict(self, A):
    # must return (samples N, h, w, 2)
    res = self.network.predict(numpy.array([A]))[0]
    return res

  def freeze(self):
    for layer in self.network.layers:
      layer.trainable = False
    return
  
  def compressed(self):
    return self.network.get_layer(name='middle').output

####################################
class CTrainingParameters:
  def __init__(self):
    return
  
  def loss(self):
    dice = MulticlassDiceLoss(weights=[1., 1.])
    def calc(y_true, y_pred):
      dloss = dice(
        y_true,
        keras.backend.pow(y_pred, 5) # push "down" predictions
      )
      return dloss
    return calc

  def DataGenerator(self, generator):
    return generator
