import win32api
import cv2
import time
import os

from .CBotDebugState import CBotDebugState
from .extractGameMap import extractGameMap
from .CNavigator import CNavigator

class CBot:
  def __init__(self, logger, navigator=None):
    self.logger = logger
    self.navigator = navigator if navigator else CNavigator()
    pass
  
  def isActive(self):
    return (win32api.GetAsyncKeyState(ord('Q')) & 1) == 0
  
  def process(self, screenshot):
    actions = []
    debug = CBotDebugState(screenshot, self.logger)
    
    if (win32api.GetAsyncKeyState(ord('A')) & 1) == 1:
      self.saveScreenshot(screenshot)
    
    # TODO: update map only in IDLE state
    mapMask, _ = extractGameMap(screenshot, returnSource=False)
    debug.put('map mask', mapMask)
    self.navigator.update(mapMask)
    
    # TODO: return action for exploring map
    
    return (actions, debug)
  
  def saveScreenshot(self, screenshot):
    os.makedirs("screenshots", exist_ok=True)
    fname = 'screenshots/%d.jpg' % time.time_ns()
    cv2.imwrite(fname, screenshot)
    self.logger.info('Screenshot saved to %s' % fname)
    return