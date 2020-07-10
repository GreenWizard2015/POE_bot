import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
from random import Random
from glob import glob

# TODO: CDataGenerator must work only with squared regions/input of constant size 
""" select single image per batch and return random crop of inputDims """
class CDataGenerator(Sequence):
  def __init__(self, folder, dims, batchSize, batchesPerEpoch, seed, usePadding=True):
    self._usePadding = usePadding
    self._batchSize = batchSize
    self._dims = np.array(dims)
    self._batchesPerEpoch = batchesPerEpoch
    self._random = Random(seed)
    self._epochBatches = None
    
    self._images = [
      (
        self._loadInput(f), 
        self._loadMasks(
          str(f).replace('_input.jpg', '_walls.jpg'),
          str(f).replace('_input.jpg', '_unknown.jpg')
        ),
        [ None ] # placeholder for distribution
      ) for f in glob('%s/*_input.jpg' % folder)
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
    sampleInput, masks, distribution = data
    sampleWalls, sampleUnknown = masks
    return (sampleInput, sampleWalls, sampleUnknown, distribution[0])

  def _randomPoint(self, w, h, distribution):
    if distribution is not None:
      x = self._random.choices(np.arange(0, w), cum_weights=distribution[0], k=1)[0]
      y = self._random.choices(np.arange(0, h), cum_weights=distribution[1], k=1)[0]
    else:
      x = self._random.randint(0, w)
      y = self._random.randint(0, h)
    return (x, y)
    
  def _generateCrops(self, dims, N=None, distribution=None):
    N = N if N else self._batchSize
    cw, ch = self._dims
    w, h = np.array(dims) - self._dims
    # always include original (0, 0)
    originPt = self._dims // 2
    crops = [(*originPt, *(originPt + self._dims))]
    for _ in range(2 * N):
      x, y = self._randomPoint(w, h, distribution)
      crop = (x, y, x + cw, y + ch)
      if crop not in crops: # only unique crops
        crops.append(crop)
    # fill by random crops, if there is a space
    while len(crops) < N:
      x, y = self._randomPoint(w, h, distribution)
      crops.append((x, y, x + cw, y + ch))
    return crops
    
  def __getitem__(self, index):
    sampleInput, sampleWalls, sampleUnknown, distribution = self._batchData( self._epochBatches[index] )
    crops = self._generateCrops(sampleInput.shape[:2], distribution=distribution)
    return (
      self._generate_X(sampleInput, crops),
      self._generate_y(sampleWalls, sampleUnknown, crops)
    )

  def _generate_X(self, img, crops):
    X = np.empty((len(crops), *self._dims, 3))

    for i, (x1, y1, x2, y2) in enumerate(crops):
      X[i,] = img[x1:x2, y1:y2]

    return X

  def _generate_y(self, imgWalls, imgUnknown, crops):
    ######
    # background = np.where(0 < (imgWalls + imgUnknown), 0, 1).astype(np.float32)
    # always ignored by loss functions, just placeholder
    background = np.zeros(imgWalls.shape, np.float32)
    
    mask = np.array([background, imgWalls, imgUnknown])
    y = np.empty((len(crops), mask.shape[0], *self._dims))
    for i, (x1, y1, x2, y2) in enumerate(crops):
      y[i,] = mask[:, x1:x2, y1:y2]

    return y

  def forgetWeakness(self):
    for sample in self._images:
      sample[2][0] = None # remove old distribution

  def learnWeakness(self, network, topK, regionsN, trueAdapter=None):
    '''
      Very dark magic, forbidden from the first days of the Universe.
      We trying to find out the weaknesses of the network and hit them ofter.
      Who knows, maybe network become smarter or just gave up O.o
    '''
    trueAdapter = trueAdapter if trueAdapter else lambda x: x
    self.forgetWeakness()
    cw, ch = self._dims
    #############
    def gaussian(x, mu, sig):
      return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    normDist = [gaussian(np.linspace(-1, 1, n * 2), mu=0, sig=.5) / topK for n in self._dims]
    #############
    def errorsDistribution(regions, dims):
      distribution = np.array(
        [np.pad(np.ones(n), cw//2, mode='constant', constant_values=1) for n in dims]
      )

      for _, (x1, y1, x2, y2) in regions:
        xc = (x1 + x2 + cw) // 2
        yc = (y1 + y2 + ch) // 2

        distribution[0, (xc - cw):(xc + cw)] += normDist[0]
        distribution[1, (yc - ch):(yc + ch)] += normDist[1]

      # unpad
      distribution = np.array([ distribution[0, (cw // 2):(-cw // 2)], distribution[1, (ch // 2):(-ch // 2)] ])
      # crop buttom-right
      distribution = distribution[:, :-ch]
      return distribution
    #############
    samples = self._random.sample(
      self._images,
      k=min(( self._batchesPerEpoch // 3, len(self._images) ))
    )
    
    for sample in samples:
      sampleInput, sampleWalls, sampleUnknown, _ = self._batchData(sample)
      crops = self._generateCrops(sampleInput.shape[:2], N=regionsN, distribution=None)
      
      predictions = network.predict(self._generate_X(sampleInput, crops))
      Y = trueAdapter(self._generate_y(sampleWalls, sampleUnknown, crops))
      losses = [
        (K.eval(network.loss(
          np.array([y_true]).astype(np.float32),
          np.array([y_pred]).astype(np.float32)
        )), y_pred) for y_true, y_pred in zip(Y, predictions)
      ]
      
      worst = sorted(zip(losses, crops), key=lambda x: x[0][0], reverse=True)[:5] 
      sample[2][0] = np.cumsum(
        errorsDistribution(worst, np.array(sampleInput.shape[:2])),
        axis=-1
      )

    return
  
  ###########################
  def _addPadding(self, images):
    if not self._usePadding: return images

    pw, ph = self._dims // 2
    return [
      cv2.copyMakeBorder(img, ph, ph, pw, pw, cv2.BORDER_REFLECT) for img in images
    ]

  def _loadInput(self, src):
    sample = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2Lab) / 255 # normalize
    return self._addPadding([sample])[0]

  def _loadMasks(self, srcWalls, srcUnknown):
    imgWalls, imgUnknown = self._addPadding([
      cv2.imread(srcWalls, cv2.IMREAD_GRAYSCALE),
      cv2.imread(srcUnknown, cv2.IMREAD_GRAYSCALE)
    ])
    
    # preprocessing
    #  Fill small holes
    cv2.GaussianBlur(imgUnknown, (3, 3), 0)
    imgWalls = np.where(80 < imgWalls, 1, 0).astype(np.float32)
    imgUnknown = np.where(80 < imgUnknown, 1, 0).astype(np.float32)
    
    imgWalls[np.where(0 < cv2.GaussianBlur(imgUnknown.copy(), (3, 3), 0))] = 0 # just to be sure
    
    return [imgWalls, imgUnknown]