import win32api
import cv2
import time
import os
import enum
import numpy as np

from .CBotDebugState import CBotDebugState
from .extractGameMap import extractGameMap
from .CNavigator import CNavigator

from CMinimapRecognizer import CMinimapRecognizer

class BotState(enum.Enum):
  IDLE = 0
  MOVING = 1
  BATTLE = 2
  
class CBot:
  def __init__(self, logger, navigator=None):
    self.logger = logger
    self._minimapRecognizer = CMinimapRecognizer()
    self.navigator = navigator if navigator else CNavigator()
    self.state = BotState.IDLE 
    pass
  
  def isActive(self):
    return (win32api.GetAsyncKeyState(ord('Q')) & 1) == 0
  
  def process(self, screenshot):
    debug = CBotDebugState(screenshot, self.logger)
    
    if (win32api.GetAsyncKeyState(ord('A')) & 1) == 1:
      self.saveScreenshot(screenshot)
    
    if BotState.IDLE == self.state:
      return self._onIdle(screenshot, debug)
    
    if BotState.MOVING == self.state:
      return self._onMoving(screenshot, debug)
      
    return ([], debug)
  
  def _onIdle(self, screenshot, debug):
    # temporally code for collecting data
    dumpMinimap = (win32api.GetAsyncKeyState(ord('T')) & 1) == 1

    mapImg = extractGameMap(screenshot)
    mapMask = self._minimapRecognizer.process(mapImg)
    if dumpMinimap:
      os.makedirs("minimap", exist_ok=True)
      fname = 'minimap/%d_%%s.jpg' % time.time_ns()
      cv2.imwrite(fname % 'input', mapImg)
      cv2.imwrite(fname % 'mask', mapMask)

    debug.put('map mask', mapMask)
    self.navigator.update(mapMask)
    
    # TODO: return action for exploring map
    return ([], debug)

  def _onMoving(self, screenshot, debug):
    # TODO: Find a way to detect when moving is ended
    return ([], debug)
    
  def saveScreenshot(self, screenshot):
    os.makedirs("screenshots", exist_ok=True)
    fname = 'screenshots/%d.jpg' % time.time_ns()
    cv2.imwrite(fname, screenshot)
    self.logger.info('Screenshot saved to %s' % fname)
    return