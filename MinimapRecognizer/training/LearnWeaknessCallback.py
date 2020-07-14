import tensorflow.keras as keras
import math

class LearnWeaknessCallback(keras.callbacks.Callback):
  def __init__(self, model, learners, patience, cooldown, rest, topK, regionsN):
    self._model = model
    self._learners = learners
    self._patience = patience
    self._cooldown = cooldown
    self._rest = cooldown - rest
    
    self._best = math.inf
    self._lastBreakdown = 0
    self._lastLW = 0
    self._weaknessActive = False

    self._topK = topK
    self._regionsN = regionsN
    pass
  
  def on_epoch_end(self, epoch, logs=None):
    if self._weaknessActive: print('Exploiting weakness.')
    
    loss = logs['val_loss'] 
    if loss < self._best:
      self._best = loss
      self._lastBreakdown = epoch
      self._lastLW = 0
    
    if (epoch - self._lastBreakdown) <= self._patience:
      if self._weaknessActive and (self._cooldown < (epoch - self._lastBreakdown)):
        self.deactivate()
      return
    
    if (epoch - self._lastLW) <= self._cooldown:
      if self._rest <= (epoch - self._lastLW):
        self.deactivate()
      return
    
    print('Start learning weakness.')
    for l in self._learners:
      l.learnWeakness(self._model.network, topK=self._topK, regionsN=self._regionsN)
    self._weaknessActive = True
    print('Done learning weakness.')
    
    self._lastLW = epoch
    return
  
  def deactivate(self):
    if self._weaknessActive:
      self._weaknessActive = False
      for l in self._learners:
        l.forgetWeakness()
    return