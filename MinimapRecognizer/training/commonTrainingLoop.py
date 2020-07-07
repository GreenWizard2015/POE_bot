from training.DataGenerator import CDataGenerator
from training.NNDebugCallback import NNDebugCallback
import tensorflow.keras as keras
import time
import cv2
from training.LearnWeaknessCallback import LearnWeaknessCallback

def commonTrainingLoop(model, batch_size, batch_per_epoch, batch_per_validation):
  if isinstance(batch_per_validation, float):
    if not batch_per_validation.is_integer():
      batch_per_validation = int(batch_per_epoch * batch_per_validation)

  params = model.trainingParams()
  
  model.network.compile(
    optimizer=keras.optimizers.Adam(lr=0.001),
    loss=params.loss(),
    metrics=[]
  )
  
  seed = time.time_ns() # kinda random seed
  dims = model.input_shape[:2]
  
  trainGenerator = params.DataGenerator(
    CDataGenerator(
      'dataset/train', dims=dims, 
      batchSize=batch_size, batchesPerEpoch=batch_per_epoch, seed=seed
    )
  )
  
  validGenerator = params.DataGenerator(
    CDataGenerator(
      'dataset/validation', dims=dims,
      batchSize=batch_size, batchesPerEpoch=batch_per_validation, seed=seed
    )
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
      ),
      NNDebugCallback(
        model=model,
        saveFreq=-1,
        dstFolder='debug/%d' % time.time_ns(),
        inputs=(
          cv2.imread('debug/src_input.jpg'),
          cv2.imread('debug/src_walls.jpg', cv2.IMREAD_GRAYSCALE),
          cv2.imread('debug/src_unknown.jpg', cv2.IMREAD_GRAYSCALE),
        )
      ),
      LearnWeaknessCallback(
        model=model, learners=[trainGenerator],
        patience=50, cooldown=50, rest=20
      )
    ],
    epochs=1000000 # we use EarlyStopping, so just a big number
  )