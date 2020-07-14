import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import hashlib
 
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
 
def model_hash(model):
  stringlist = []
  model.summary(print_fn=lambda x: stringlist.append(x))
  return hashlib.md5("".join(stringlist).encode('utf8')).hexdigest()

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