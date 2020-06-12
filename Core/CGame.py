from pywinauto import handleprops
import win32gui
from Core.grab_screenshot import grab_screen

class CGame:
  def __init__(self, logger):
    self.logger = logger
    self._lastHWND = None
    self._lastWndIsGame = False
    pass
  
  def isActive(self):
    hwnd = win32gui.GetForegroundWindow()
    if not self._lastHWND == hwnd:
      self._lastHWND = hwnd
      self._lastWndIsGame = 0 <= handleprops.text(hwnd).lower().find('path of exile')

    return self._lastWndIsGame
  
  def screenshot(self):
    return grab_screen(self._lastHWND, 'BGR')
  
  def execute(self, actions):
    # @todo: send actions to game window
    return False