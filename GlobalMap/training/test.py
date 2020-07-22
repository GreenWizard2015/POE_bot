from GlobalMap.training.CDataGenerator import CDataGenerator
import cv2
gen = CDataGenerator(
  folder='dataset/train/area 0/',
  batchSize=5,
  batchesPerEpoch=1,
  seed=0,
  bigMapSize=1024,
  smallMapSize=256
)

while True:
  gen.on_epoch_end()
  a = gen[0][0]
  for (img,) in a:
    cv2.imshow('', img)
    cv2.waitKey()
  break