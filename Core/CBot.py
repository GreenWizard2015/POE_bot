import win32api
from .CBotDebugState import CBotDebugState

class CBot:
  def __init__(self, logger):
    self.logger = logger
    pass
  
  def isActive(self):
    return (win32api.GetAsyncKeyState(ord('Q')) & 1) == 0
  
  def process(self, screenshot):
    actions = []
    debug = CBotDebugState(screenshot, self.logger)
    return (actions, debug)