#!/usr/bin/env python
# -*- coding: utf-8 -*-

# limit GPU memory usage or TF eat it all
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
  gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
)

import logging
from Core.CBot import CBot
from Core.CGame import CGame
import cv2

logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
  debugOutput = ['map mask walls', 'map mask unknown']
  bot = CBot(logger)
  game = CGame(logger)
  while bot.isActive():
    cv2.waitKey(1)
    if not game.isActive(): continue
    
    screenshot = game.screenshot()
    actions, debugInfo = bot.process(screenshot)
    game.execute(actions)
    debugInfo.show(debugOutput)
    pass
  
  cv2.destroyAllWindows()
  
if __name__ == '__main__':
  main()
