from GlobalMap.training.CGlobalMapGenerator import CDataGenerator
import cv2
import numpy as np

gen = CDataGenerator(
  folder='dataset/',
  batchSize=5,
  batchesPerEpoch=1,
  seed=0,
  bigMapSize=1024,
  minCommonPoints=20,
  minInnerPoints=80,
  smallMapSize=256
)

while True:
  gen.on_epoch_end()
  samples = gen[0]
  for (img, crop, posX_OHE, posY_OHE) in zip(*samples):
    pos = (np.argmax(posY_OHE, axis=-1), np.argmax(posX_OHE, axis=-1))
    img = cv2.cvtColor(img.astype(np.uint8) * 128, cv2.COLOR_GRAY2BGR) 
    
    A = tuple((np.array(pos) - 256 // 2).astype(np.uint32))
    B = tuple((np.array(pos) + 256 // 2).astype(np.uint32))

    cv2.rectangle(img, A, B, (0, 255, 0), 1)
    img[A[1]:B[1], A[0]:B[0], 0] += crop.astype(np.uint8) * 128
    cv2.imshow('', img)
    cv2.waitKey()
