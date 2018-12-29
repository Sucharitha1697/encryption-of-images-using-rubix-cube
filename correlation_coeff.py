import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from scipy import misc
from PIL import Image
from scipy import fftpack
import matplotlib.pylab as pylab
import matplotlib.image as mpimg
from skimage.feature import greycomatrix
from enum import Enum
from math import sqrt
import random
import sys
import copy
import math
import skimage
from skimage import measure
from numpy import unique
from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy

value_of_x=0
value_of_y=0


def co1(height,width,pixels):
    value=0
    for pixel_coordinate_of_y in range(0, height):
        for pixel_coordinate_of_x in range(0, width):
            if pixel_coordinate_of_x+1==width:
                break
            value=int(pixels[pixel_coordinate_of_x,pixel_coordinate_of_y])*int(pixels[pixel_coordinate_of_x+1,pixel_coordinate_of_y])+value


    return value*height*width

def co2(height,width,pixels):
   global value_of_y
   global value_of_x
   for pixel_coordinate_of_y in range(0, height):
        for pixel_coordinate_of_x in range(0, width):
            if pixel_coordinate_of_x+1==width:
                break
            value_of_x=int(pixels[pixel_coordinate_of_x,pixel_coordinate_of_y])+value_of_x
            value_of_y=int(pixels[pixel_coordinate_of_x+1,pixel_coordinate_of_y])+value_of_y

   return value_of_x*value_of_y


def co3(height,width,pixels):
    value=0
    for pixel_coordinate_of_y in range(0, height):
        for pixel_coordinate_of_x in range(0, width):
            value=int(pixels[pixel_coordinate_of_x,pixel_coordinate_of_y])**2 +value

    xy=abs(int(value*height*width)-int(value_of_x**2))
    return  xy

def co4(height,width,pixels):
    value=0
    for pixel_coordinate_of_y in range(0, height):
        for pixel_coordinate_of_x in range(0, width):
            if pixel_coordinate_of_x+1==width:
                break
            value=int(pixels[pixel_coordinate_of_x+1,pixel_coordinate_of_y]**2)+value

    xy=abs(int(value*height*width)-int(value_of_y**2))
    return xy

def corr(img):
    global value_of_y
    global value_of_x
    width,height=np.shape(img)
    pixels =img
    #width, height =img.size
    value_of_y = 0
    value_of_x = 0
    cr=abs(co1(height,width,pixels)-co2(height,width,pixels)) / sqrt(co3(height,width,pixels)*co4(height,width,pixels))
    return cr
