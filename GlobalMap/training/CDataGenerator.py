import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from random import Random
from glob import glob
from math import floor

'''
  ---------------------------------
  -                               -
  -   X------------------------|  -
  -   |       |                |  -
  -   |   B********************|  -
  -   |   *   |                |  -
  -   |---*---S                |  -
  -   |   *                    |  -
  -   |------------------------|  -
  -                               -
  ---------------------------------
  
  - Take masks
  - Add padding (wh -> 3*wh)
  - Take a random region X and split into S and B
    - X must contain >xMin 'pixels' 
    - area SB must contain >overlapMin 'pixels'
  
  - Return (big map), (small map), (x, y of S in B)
'''
class CDataGenerator(Sequence):
  def __init__(self, folder, batchSize, batchesPerEpoch, seed, bigMapSize, smallMapSize=256):
    self._batchSize = batchSize
    self._batchesPerEpoch = batchesPerEpoch
    self._random = Random(seed)
    
    self._bigMapSize = bigMapSize
    self._smallMapSize = smallMapSize
    
    self._epochBatches = None
    self._images = [
      ( 
        self._loadMasks(
          str(f).replace('_input.jpg', '_walls.jpg'),
          str(f).replace('_input.jpg', '_unknown.jpg')
        ),
      ) for f in glob('%s/**/*_input.jpg' % folder, recursive=True)
    ]
    self.on_epoch_end()
    return

  def __len__(self):
    return self._batchesPerEpoch

  def on_epoch_end(self):
    """Updates after each epoch"""
    self._epochBatches = self._random.choices(self._images, k=self._batchesPerEpoch)
    return
  
  def _batchData(self, data):
    sampleWalls, sampleUnknown = data[0]
    return (sampleWalls, sampleUnknown)
    
  def _generateCrops(self, img, N=None):
    N = N if N else self._batchSize
    w = img.shape[0] - self._smallMapSize
    crops = []
    while len(crops) < N:
      overlapArea = self._random.randint(10, self._smallMapSize - 5)
      
      X_w = self._random.randint(overlapArea + self._smallMapSize, 2 * self._smallMapSize)
      X_h = self._random.randint(overlapArea + self._smallMapSize, 2 * self._smallMapSize)
      
      # LT of X
      X_x = self._random.randint(0, w - X_w)
      X_y = self._random.randint(0, w - X_h)
      
      # top, bottom, left, right
      S_xTable = (
        self._random.randint(0, X_w - self._smallMapSize),
        self._random.randint(0, X_w - self._smallMapSize),
        0,
        X_w - self._smallMapSize
      )
      S_yTable = (
        0,
        X_h - self._smallMapSize,
        self._random.randint(0, X_h - self._smallMapSize),
        self._random.randint(0, X_h - self._smallMapSize)
      )
      
      side = self._random.randint(0, 3)
      S_x = S_xTable[side]
      S_y = S_yTable[side]
      
      B_x = self._random.randint(0, self._bigMapSize - X_w)
      B_y = self._random.randint(0, self._bigMapSize - X_h)

      crops.append((
        (X_x, X_y, X_w, X_h), # main region
        (S_x, S_y), # small
        (B_x, B_y), # big
        overlapArea
      ))

    return crops
    
  def __getitem__(self, index):
    sampleWalls, _ = self._batchData( self._epochBatches[index] )
    crops = self._generateCrops(sampleWalls)
    return (
      self._generate_XA(sampleWalls, crops),
      None, # self._generate_XB(sampleWalls, crops),
      None # self._generate_y(crops)
    )

  def _generate_XA(self, img, crops):
    X = np.zeros((len(crops), 1, self._bigMapSize, self._bigMapSize))

    for i, crop in enumerate(crops):
      X_x, X_y, X_w, X_h = crop[0] # main region
      S_x, S_y = crop[1] # small
      B_x, B_y = crop[2] # big
      overlapArea = crop[3]
      
      reg = img[X_x:X_x+X_w, X_y:X_y+X_h]
      reg[S_x:S_x+self._smallMapSize-overlapArea, S_y:S_y+self._smallMapSize-overlapArea] = 0
      X[i, 0, B_x:B_x+X_w, B_y:B_y+X_h] = reg

    return X

  def _generate_XB(self, img, crops):
    X = np.zeros((len(crops), 2, self._smallMapSize, self._smallMapSize))

    for i, (x1, y1, _, _) in enumerate(crops):
      X[i,] = 1

    return X

  def _generate_y(self, crops):
    Y = np.zeros((len(crops), 2, self._bigMapSize))

    for i, (x1, y1, _, _) in enumerate(crops):
      Y[i, x1, y1] = 1

    return Y
  ###########################
  def _addPadding(self, images):
    pw, ph = self._smallMapSize, self._smallMapSize
    return [
      cv2.copyMakeBorder(img, ph, ph, pw, pw, cv2.BORDER_REFLECT) for img in images
    ]

  def _loadMasks(self, srcWalls, srcUnknown):
    imgWalls, imgUnknown = self._addPadding([
      cv2.imread(srcWalls, cv2.IMREAD_GRAYSCALE),
      cv2.imread(srcUnknown, cv2.IMREAD_GRAYSCALE)
    ])
    
    imgWalls = np.where(80 < imgWalls, 1, 0).astype(np.float32)
    imgUnknown = np.where(80 < imgUnknown, 1, 0).astype(np.float32)
    imgWalls[np.where(0 < cv2.GaussianBlur(imgUnknown.copy(), (3, 3), 0))] = 0 # just to be sure
    
    imgUnknown = None
    return [imgWalls, imgUnknown]