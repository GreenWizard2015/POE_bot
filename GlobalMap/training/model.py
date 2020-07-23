import tensorflow.keras as keras
import tensorflow.keras.layers as layers
 
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

def GMNetwork(input_shape_big, input_shape_small):
  sz = input_shape_big[0]
  inputA = layers.Input(shape=input_shape_big)
  inputB = layers.Input(shape=input_shape_small)
  
  scaleFactor = sz // input_shape_small[0]

  res = layers.Concatenate()([
    layers.Conv2D(3, 1, activation='relu', padding='same')(inputA),
    layers.Convolution2DTranspose(3, (scaleFactor, scaleFactor), strides=scaleFactor)(inputB)
  ])
  
  convA, res = downsamplingBlockWithLink(res, 3, 4)
  convB, res = downsamplingBlockWithLink(res, 3, 8)
  convC, res = downsamplingBlockWithLink(res, 3, 12)
  convD, res = downsamplingBlockWithLink(res, 3, 14)
  
  res = convBlock(res, 3, 16)
  
  res = upsamplingBlock(res, convD, 3, 14)
  res = upsamplingBlock(res, convC, 3, 12)
  res = upsamplingBlock(res, convB, 3, 8)
  res = upsamplingBlock(res, convA, 3, 4)
  res = layers.Convolution2D(4, 1, padding="same", activation="softmax")(res)
  
  res = layers.Convolution2D(1, 1, padding="same")(res)
  res = layers.Convolution2D(1, 1, padding="same")(res)
  res = layers.Convolution2D(1, 1, padding="same", activation="softmax")(res)
  
  return keras.Model(
    inputs=(inputA, inputB),
    outputs=res
  )