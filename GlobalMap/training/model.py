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
  blockA = res = layers.Convolution2D(8, (3, 3), padding="same", activation="relu", strides=2, name="Block_A")(res)
  
  # block B  
  res = convBlock(res, sz=3, filters=32, strides=1)
  blockB = res = layers.Convolution2D(16, (3, 3), padding="same", activation="relu", strides=2, name="Block_B")(res)
  
  # block C
  res = convBlock(res, sz=3, filters=64, strides=1)
  blockC = res = layers.Convolution2D(32, (3, 3), padding="same", activation="relu", strides=2, name="Block_C")(res)
  
  # block D
  res = convBlock(res, sz=3, filters=128, strides=1)
  blockD = res = layers.Convolution2D(64, (3, 3), padding="same", activation="relu", strides=2, name="Block_D")(res)
  
  # block E
  res = convBlock(res, sz=3, filters=256, strides=1)
  blockE = res = layers.Convolution2D(128, (3, 3), padding="same", activation="relu", strides=2, name="Block_E")(res)
  #######################
  blockA = convBlock(blockA, sz=4, filters=64, strides=4)
  blockA = convBlock(blockA, sz=4, filters=64, strides=4)
  blockA = layers.Convolution2D(64, (4, 4), padding="same", activation="relu", strides=4, name="Block_A_out")(blockA)
  
  blockB = convBlock(blockB, sz=4, filters=64, strides=4)
  blockB = convBlock(blockB, sz=4, filters=64, strides=4)
  blockB = layers.Convolution2D(64, (2, 2), padding="same", activation="relu", strides=2, name="Block_B_out")(blockB)
  
  blockC = convBlock(blockC, sz=4, filters=64, strides=4)
  blockC = layers.Convolution2D(64, (4, 4), padding="same", activation="relu", strides=4, name="Block_C_out")(blockC)
  
  blockD = convBlock(blockD, sz=4, filters=64, strides=4)
  blockD = layers.Convolution2D(64, (4, 4), padding="same", activation="relu", strides=4, name="Block_D_out")(blockD)
  
  blockE = convBlock(blockE, sz=4, filters=64, strides=4)
  blockE = layers.Convolution2D(64, (4, 4), padding="same", activation="relu", strides=4, name="Block_E_out")(blockE)

  res = layers.Concatenate(axis=-1)([blockA, blockB, blockC, blockD, blockE])
  res = layers.Dense(2, activation='tanh')(
    layers.Flatten()(res)
  )
  return keras.Model(
    inputs=(inputA, inputB),
    outputs=res
  )