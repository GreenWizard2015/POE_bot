import numpy as np
from tensorflow.keras.utils import Sequence
from random import Random

class CNoiseInputGenerator(Sequence):
  def __init__(self, gen, 
    seed,
    noiseAdded, noiseVanished
  ):
    self._gen = gen
    self._random = Random(seed)
    
    self._minAdded = noiseAdded
    self._minVanished = noiseVanished
    
    self.on_epoch_end()
    return

  def __len__(self):
    return len(self._gen)

  def on_epoch_end(self):
    """Updates after each epoch"""
    self._gen.on_epoch_end()
    return
  
  def _addNoise(self, img):
    res = img
    added = vanished = N = 0
    while (N < 64) and ((added < self._minAdded) or (vanished < self._minVanished)):
      x = self._random.randint(0, img.shape[0] - 3)
      y = self._random.randint(0, img.shape[0] - 3)
      area = res[x:x+3, y:y+3, 0]
      total = np.sum(area)
      
      N += 1
      if (0 < area[1, 1]) and (vanished < self._minVanished) and (3 < total):
        res[x + 1, y + 1, 0] = 0 
        vanished += 1
        N = 0
        continue
      
      if (area[1, 1] < 1) and (added < self._minAdded) and (total < 5):
        res[x + 1, y + 1, 0] = 1
        added += 1
        N = 0
        continue
    return res

  def __getitem__(self, index):
    A, res = self._gen[index]
    return (
      np.array([self._addNoise(x) for x in A]),
      np.array(res)
    )