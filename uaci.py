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

def uaci(pixel1,pixel2):
    width,height=np.shape(pixel1)
    value=0
    for y in range(0,height):
        for x in range(0,width):
             value=(abs(int(pixel1[x,y])-int(pixel2[x,y]))/255)+value
            

    value=(value*100)/(width*height)
    return value
