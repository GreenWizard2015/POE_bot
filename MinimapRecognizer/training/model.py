import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from Utils.NNBlocks import *

def MRNetwork(input_shape):
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
    outputs=layers.Conv2D(3, 1, activation='softmax', padding='same')(res)
  )