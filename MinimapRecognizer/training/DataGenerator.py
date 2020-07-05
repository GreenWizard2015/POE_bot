import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from random import Random
from glob import glob

""" select single image per batch and return random crop of inputDims """
class CDataGenerator(Sequence):
  def __init__(self, folder, dims, batchSize, batchesPerEpoch, seed):
    self._images = [
      (
        str(f), str(f).replace('_input.jpg', '_walls.jpg'), str(f).replace('_input.jpg', '_unknown.jpg')
      ) for f in glob('%s/*_input.jpg' % folder)
    ]
    self._batchSize = batchSize
    self._dims = dims
    self._batchesPerEpoch = batchesPerEpoch
    self._random = Random(seed)
    self._epochBatches = None
    self.on_epoch_end()
    return 

  def on_epoch_end(self):
    """Updates after each epoch"""
    self._epochBatches = self._random.choices(self._images, k=self._batchesPerEpoch)
    return

  def __len__(self):
    return self._batchesPerEpoch

  def __getitem__(self, index):
    # TODO: Cache images/masks
    sampleInput, sampleWalls, sampleUnknown = self._epochBatches[index]
    sampleInput = cv2.cvtColor(cv2.imread(sampleInput), cv2.COLOR_BGR2Lab) / 255 # normalize
    # TODO: Generate samples on the borders of input (i.e. randint(-cw // 2, w - cw // 2) )
    cw, ch = self._dims
    w, h = np.array([sampleInput.shape[0] - cw, sampleInput.shape[1] - ch])
    crops = []
    for _ in range(self._batchSize):
      x = self._random.randint(0, w)
      y = self._random.randint(0, h) 
      crops.append((x, y, x + cw, y + ch))

    X = self._generate_X(sampleInput, crops)
    y = self._generate_y(
      cv2.imread(sampleWalls, cv2.IMREAD_GRAYSCALE),
      cv2.imread(sampleUnknown, cv2.IMREAD_GRAYSCALE),
      crops
    )

    return X, y

  def _generate_X(self, img, crops):
    X = np.empty((self._batchSize, *self._dims, 3))

    for i, (x1, y1, x2, y2) in enumerate(crops):
      X[i,] = img[x1:x2, y1:y2]

    return X

  def _generate_y(self, imgWalls, imgUnknown, crops):
    # preprocessing
    cv2.GaussianBlur(imgUnknown, (5, 5), 0)
    imgWalls = np.where(80 < imgWalls, 1, 0).astype(np.float32)
    imgUnknown = np.where(80 < imgUnknown, 1, 0).astype(np.float32)
    
    imgWalls[np.where(0 < imgUnknown)] = 0 # just to be sure
    
    ######
    # background = np.where(0 < (imgWalls + imgUnknown), 0, 1).astype(np.float32)
    # always ignored by loss functions, just placeholder
    background = np.zeros(imgWalls.shape, np.float32)
    
    mask = np.array([background, imgWalls, imgUnknown])
    y = np.empty((self._batchSize, mask.shape[0], *self._dims))
    for i, (x1, y1, x2, y2) in enumerate(crops):
      y[i,] = mask[:, x1:x2, y1:y2]

    return y
