import tensorflow.keras.backend as K

def dice_coef(weights=None):
  def calc(y_true, y_pred):
    # shapes must be (batch, classes, h, w)
    axis = [2, 3]
    intersection = K.sum(y_pred * y_true, axis=axis)
    sums = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis)
  
    # standard dice
    dice = (2. * intersection + K.epsilon()) / (sums + K.epsilon())
    
    dice = K.sum(dice * weights, axis=-1)
    # mean over all samples in the batch
    return K.mean(dice, axis=-1)
  
  return calc

def MulticlassDiceLoss(weights):
  diceCoef = dice_coef(weights)
  def calc(y_true, y_pred):
    # (batch, h, w, classes) -> (batch, classes, h, w)
    y_pred = K.permute_dimensions(y_pred, (0, 3, 1, 2))
    return 1. - diceCoef(y_true, y_pred)

  return calc
