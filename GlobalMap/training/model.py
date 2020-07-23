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
  inputA = layers.Input(shape=input_shape_big)
  inputB = layers.Input(shape=input_shape_small)
  
  scaleFactor = input_shape_big[0] // input_shape_small[0]

  res = layers.Concatenate()([
    layers.Conv2D(16, 1, activation='relu', padding='same')(inputA),
    layers.Convolution2DTranspose(16, (scaleFactor, scaleFactor), strides=scaleFactor)(inputB)
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

  sz = input_shape_big[0]
  X = layers.Flatten() (
    layers.Conv2D(1, (1, sz), strides=(1, sz), activation='softmax')(res)
  )
  Y = layers.Flatten() (
    layers.Conv2D(1, (sz, 1), strides=(sz, 1), activation='softmax')(res)
  )
  
  return keras.Model(
    inputs=(inputA, inputB),
    outputs=(X, Y)
  )