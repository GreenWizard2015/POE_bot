from pywinauto import handleprops
import win32gui
from Core.grab_screenshot import grab_screen
import numpy as np
import pywinauto

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
    for action in actions:
      if 'move' == action[0]:
        pos = action[1]
        dx, dy = (pos / np.linalg.norm(pos) * 335).astype(np.int32)
        left, top, right, buttom = win32gui.GetClientRect(self._gameHWND)
        
        x, y = ( ((left + right) // 2) + dx, ((top + buttom) // 2) + dy )
        pywinauto.mouse.click(coords=(x, y))
    return False