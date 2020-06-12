import win32api
from .CBotDebugState import CBotDebugState
import cv2
import time
import os

class CBot:
  def __init__(self, logger):
    self.logger = logger
    pass
  
  def isActive(self):
    return (win32api.GetAsyncKeyState(ord('Q')) & 1) == 0
  
  def process(self, screenshot):
    actions = []
    debug = CBotDebugState(screenshot, self.logger)
    
    if (win32api.GetAsyncKeyState(ord('A')) & 1) == 1:
      self.saveScreenshot(screenshot)
      
    return (actions, debug)
  
  def saveScreenshot(self, screenshot):
    os.makedirs("screenshots", exist_ok=True)
    fname = 'screenshots/%d.jpg' % time.time_ns()
    cv2.imwrite(fname, screenshot)
    self.logger.info('Screenshot saved to %s' % fname)
    return