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

def GMNetwork(input_shape):
  inputA = layers.Input(shape=input_shape)
  inputB = layers.Input(shape=input_shape)
  
  res = layers.Concatenate(axis=-1)([inputA, inputB])
  # aka ShiftNet (https://www.mdpi.com/1424-8220/19/23/5310/htm)
  # block A  
  res = convBlock(res, sz=3, filters=16, strides=2)
  blockA = res = convBlock(res, sz=3, filters=16, strides=2)
  
  # block B  
  res = convBlock(res, sz=3, filters=32, strides=1)
  blockB = res = convBlock(res, sz=3, filters=32, strides=2)
  
  # block C
  res = convBlock(res, sz=3, filters=64, strides=1)
  blockC = res = convBlock(res, sz=3, filters=64, strides=2)
  #######################
  blockA = convBlock(blockA, sz=4, filters=64, strides=4)
  blockA = convBlock(blockA, sz=4, filters=64, strides=4)
  blockA = convBlock(blockA, sz=4, filters=64, strides=4)
  
  blockB = convBlock(blockB, sz=4, filters=64, strides=4)
  blockB = convBlock(blockB, sz=4, filters=64, strides=4)
  blockB = convBlock(blockB, sz=2, filters=64, strides=2)
  
  blockC = convBlock(blockC, sz=4, filters=64, strides=4)
  blockC = convBlock(blockC, sz=4, filters=64, strides=4)

  res = layers.Concatenate(axis=-1)([blockA, blockB, blockC])
  res = layers.Dense(2, activation='tanh')(
    layers.Flatten()(res)
  )
  return keras.Model(
    inputs=(inputA, inputB),
    outputs=res
  )