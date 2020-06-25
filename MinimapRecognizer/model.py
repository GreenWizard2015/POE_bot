import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import hashlib

def convBlock(input, sz, filters=2):
  conv_1 = layers.Convolution2D(filters, (sz, sz), padding="same", activation="relu")(input)
  conv_1 = layers.Dropout(0.1)(conv_1)
  conv_1 = layers.BatchNormalization()(conv_1)
  return conv_1

def upsamplingBlock(input, shortcut, sz, filters=2):
  input = layers.UpSampling2D((2, 2), interpolation="nearest")(input)
  concatenated = layers.Add()([input, shortcut])
  
  return convBlock(concatenated, sz, filters)

def MRModel(input_shape, num_classes, weights = None):
  res = inputs = layers.Input(shape=input_shape)
  
  convA = res = convBlock(res, 3)
  res = layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(res)
  
  res = convBlock(res, 3)
  res = upsamplingBlock(res, convA, 3)

  # output
  res = layers.Conv2D(num_classes, (3, 3), activation='sigmoid', padding='same')(res)
  
  model = keras.Model(inputs=inputs, outputs=res)
  if weights:
    model.set_weights(weights)
  return model

def model_hash(model):
  stringlist = []
  model.summary(print_fn=lambda x: stringlist.append(x))
  return hashlib.md5("".join(stringlist).encode('utf8')).hexdigest()