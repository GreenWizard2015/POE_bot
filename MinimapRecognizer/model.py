import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import hashlib
 
def convBlock(prev, sz, filters):
  conv_1 = layers.Convolution2D(filters, (sz, sz), padding="same", activation="relu")(prev)
  conv_1 = layers.Dropout(0.1)(conv_1)
  conv_1 = layers.BatchNormalization()(conv_1)
  return conv_1
 
def upsamplingBlock(prev, shortcut, sz, filters):
  prev = layers.UpSampling2D((2, 2), interpolation="nearest")(prev)
  concatenated = layers.Concatenate()([prev, shortcut])
   
  return convBlock(concatenated, sz, filters)

# TODO: Convert into class
def MRModel(input_shape, num_classes):
  res = inputs = layers.Input(shape=input_shape)
  
  convA = res = convBlock(res, 3, 4)
  res = layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(res)
  
  convB = res = convBlock(res, 3, 4)
  res = layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(res)
  
  convC = res = convBlock(res, 3, 4)
  res = layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(res)
  
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

def StackedMRModel(input_shape, models):
  inputs = layers.Input(shape=input_shape)
  ########
  for model in models:
    model.trainable = False
    for layer in model.layers: layer.trainable = False
  modelsOut = [model(inputs) for model in models]
  
  res = layers.Concatenate()([inputs, *modelsOut])
  ########
  convA = res = convBlock(res, 3, 4)
  res = layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(res)
  
  convB = res = convBlock(res, 3, 4)
  res = layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(res)
  
  convC = res = convBlock(res, 3, 4)
  res = layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(res)
  
  res = convBlock(res, 3, 4)
  
  res = upsamplingBlock(res, convC, 3, 4)
  res = upsamplingBlock(res, convB, 3, 4)
  res = upsamplingBlock(res, convA, 3, 4)

  return (keras.Model(
    inputs=inputs,
    outputs=layers.Conv2D(3, 1, activation='softmax', padding='same')(res)
  ), *models)
 