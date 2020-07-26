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

def GMNetworkHead(input_shape):
  input = res = layers.Input(shape=input_shape)
  
  convA, res = downsamplingBlockWithLink(res, 3, 8)
  convB, res = downsamplingBlockWithLink(res, 3, 8)

  res = convBlock(res, 3, 16)
  
  res = upsamplingBlock(res, convB, 3, 8)
  res = upsamplingBlock(res, convA, 3, 8)
  
  res = layers.Convolution2D(32, 7, padding="same")(res)
  
  return keras.Model(inputs=input, outputs=res)

def GMNetwork(input_shape):
  inputA = layers.Input(shape=input_shape)
  inputB = layers.Input(shape=input_shape)
  
  head = GMNetworkHead(input_shape)

  res = layers.Concatenate()([head(inputA), head(inputB)])
  
  _, res = downsamplingBlockWithLink(res, 3, 8)
  _, res = downsamplingBlockWithLink(res, 3, 8)
  _, res = downsamplingBlockWithLink(res, 3, 8)
  _, res = downsamplingBlockWithLink(res, 3, 8)
  _, res = downsamplingBlockWithLink(res, 3, 8)
  
  res = layers.Convolution2D(2, 1, activation='sigmoid')(res)
  res = layers.GlobalAveragePooling2D()(res)
  
  return keras.Model(
    inputs=(inputA, inputB),
    outputs=res
  )