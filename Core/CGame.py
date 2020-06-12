from pywinauto import handleprops
import win32gui
from Core.grab_screenshot import grab_screen

class CGame:
  def __init__(self, logger):
    self.logger = logger
    self._lastHWND = None
    self._gameHWND = None
    self._lastWndIsGame = False
    pass
  
  def isActive(self):
    hwnd = win32gui.GetForegroundWindow()
    if not self._gameHWND:
      # @todo: find game automatically      
      if 0 <= handleprops.text(hwnd).lower().find('path of exile'):
        self._gameHWND = hwnd

    return self._gameHWND
  
  def screenshot(self):
    return grab_screen(self._gameHWND, 'BGR')
  
  def execute(self, actions):
    # @todo: send actions to game window
    return False