import os
import tensorflow.keras.backend as K
from MinimapRecognizer.training.losses import MulticlassDiceLoss
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

def downsamplingBlock(prev, sz, filters):
  conv_1 = layers.Convolution2D(filters, (sz, sz), padding="same", activation="relu", strides=2)(prev)
  conv_1 = layers.Dropout(0.1)(conv_1)
  conv_1 = layers.BatchNormalization()(conv_1)
  return conv_1

def upsamplingBlock(prev, filters):
  conv_1 = layers.Convolution2DTranspose(filters, (2, 2), strides=2, padding="same")(prev)
  conv_1 = layers.Dropout(0.1)(conv_1)
  conv_1 = layers.BatchNormalization()(conv_1)
  return conv_1

def MKNetwork(input_shape):
  inputs = layers.Input(shape=input_shape)
  
  res = downsamplingBlock(inputs, sz=3, filters=16)
  res = downsamplingBlock(res, sz=3, filters=16)
  res = downsamplingBlock(res, sz=3, filters=16)
  
  res = layers.Convolution2D(8, (2, 2), padding="same", activation="sigmoid", strides=1, name="middle")(res)
  
  res = upsamplingBlock(res, filters=16)
  res = upsamplingBlock(res, filters=16)
  res = upsamplingBlock(res, filters=16)
  
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
    def calc(y_true, y_pred):
      return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
    return MulticlassDiceLoss([1])

  def DataGenerator(self, generator):
    return generator
