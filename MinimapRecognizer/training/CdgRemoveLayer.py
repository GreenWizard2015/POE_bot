import numpy as np
from tensorflow.keras.utils import Sequence

class CdgRemoveLayer(Sequence):
  def __init__(self, generator, layer):
    self._generator = generator
    self._layer = layer
    self.on_epoch_end()
    return 

  def on_epoch_end(self):
    return self._generator.on_epoch_end()

  def __len__(self):
    return len(self._generator)

  def __getitem__(self, index):
    X, y = self._generator[index]
    return X, np.delete(y, self._layer, 1)