#!/usr/bin/env python
# -*- coding: utf-8 -*-
import win32gui
import win32ui
import win32con
import cv2
import numpy

def grab_screen(hwin, color):
  left, top, right, buttom = win32gui.GetClientRect(hwin)
  height = buttom - top
  width = right - left
  
  try:
    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = numpy.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)
    
    if color.upper() == 'RGB':
        color = cv2.COLOR_BGRA2RGB
    elif color.upper() == 'GRAY':
        color = cv2.COLOR_BGRA2GRAY
    elif color.upper() == 'BGR':
        color = cv2.COLOR_BGRA2BGR

    return cv2.cvtColor(img, color)
  finally:
    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())
