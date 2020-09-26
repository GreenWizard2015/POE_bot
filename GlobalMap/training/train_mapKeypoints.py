import sys
import os
import tensorflow as tf

if 'COLAB_GPU' in os.environ:
  # fix resolve modules
  from os.path import dirname
  sys.path.append(dirname(dirname(dirname(__file__))))
else: # local GPU
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3 * 1024)]
  )

import tensorflow.keras as keras
import time



folder = lambda x: os.path.join(os.path.dirname(__file__), x)

model = CMapKeypointsModel()
model.load(only_fully_trained = False, reset=True)
model.network.summary()

batch_size=32
batch_per_epoch=32
batch_per_validation = 0.2
batch_per_validation = int(batch_per_epoch * batch_per_validation)

params = model.trainingParams()
model.network.compile(
  optimizer=keras.optimizers.Adam(lr=0.001),
  loss=params.loss(),
  metrics=[]
)

seed = time.time() # kinda random seed

trainGenerator = params.DataGenerator(
  CNoiseMapGenerator(
    folder('dataset/train'), 
    batchSize=batch_size, batchesPerEpoch=batch_per_epoch, seed=seed,
    outputMapSize=256,
    noiseAdded=64,
    noiseVanished=64
  )
)

validGenerator = params.DataGenerator(
  CNoiseMapGenerator(
    folder('dataset/validation'),
    batchSize=batch_size, batchesPerEpoch=batch_per_validation, seed=seed,
    outputMapSize=256,
    noiseAdded=64,
    noiseVanished=64
  )
)

# create folder for weights
os.makedirs(
  os.path.dirname(model.weights_file),
  exist_ok=True
)

model.network.fit(
  x=trainGenerator,
  validation_data=validGenerator,
  verbose=2,
  callbacks=[
    keras.callbacks.EarlyStopping(
      monitor='val_loss', mode='min',
      patience=250
    ),
    keras.callbacks.ModelCheckpoint(
      filepath=model.weights_file,
      save_weights_only=True,
      save_best_only=True,
      monitor='val_loss',
      mode='min',
      verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
      monitor='val_loss', factor=0.9,
      patience=50
    )
  ],
  epochs=1000000 # we use EarlyStopping, so just a big number
)