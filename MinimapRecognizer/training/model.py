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
  res = layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(link)
  return link, res
  
def upsamplingBlock(prev, shortcut, sz, filters):
  prev = layers.UpSampling2D((2, 2), interpolation="nearest")(prev)
  concatenated = layers.Concatenate()([prev, shortcut])
   
  return convBlock(concatenated, sz, filters)

def MRSubNetwork(input_shape, num_classes):
  res = inputs = layers.Input(shape=input_shape)
  
  convA, res = downsamplingBlockWithLink(res, 3, 4)
  convB, res = downsamplingBlockWithLink(res, 3, 4)
  convC, res = downsamplingBlockWithLink(res, 3, 4)
  
  res = convBlock(res, 3, 4)
  
  res = upsamplingBlock(res, convC, 3, 4)
  res = upsamplingBlock(res, convB, 3, 4)
  res = upsamplingBlock(res, convA, 3, 4)

  return keras.Model(
    inputs=inputs,
    outputs=layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(res)
  )
 
def model_hash(model):
  stringlist = []
  model.summary(print_fn=lambda x: stringlist.append(x))
  return hashlib.md5("".join(stringlist).encode('utf8')).hexdigest()

def MRNetwork(input_shape):
  inputs = layers.Input(shape=input_shape)
  ########
  models = [MRSubNetwork(input_shape, 2), MRSubNetwork(input_shape, 2)]
  modelsOut = [model(inputs) for model in models]
  
  res = layers.Concatenate()([inputs, *modelsOut])
  ########
  convA, res = downsamplingBlockWithLink(res, 3, 4)
  convB, res = downsamplingBlockWithLink(res, 3, 4)
  convC, res = downsamplingBlockWithLink(res, 3, 4)
  
  res = convBlock(res, 3, 4)
  
  res = upsamplingBlock(res, convC, 3, 4)
  res = upsamplingBlock(res, convB, 3, 4)
  res = upsamplingBlock(res, convA, 3, 4)

  return keras.Model(
    inputs=inputs,
    outputs=layers.Conv2D(3, 1, activation='softmax', padding='same')(res)
  )