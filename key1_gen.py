import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from scipy import misc
from PIL import Image
# pip install Pillow
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
#from scipy.stats import entropy as scipy_entropy
from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy

def gen1():

    r=input("input r between 2 and 4 : ")
    x=input("input x between 0 and 1 : ")
    key1=[]
    temp_key1=[]
    key1.append(float(x))
    temp_key1.append(float(x))
    temp_key1[0]=int(round(key1[0]*255))
    key1[0]='{0:08b}'.format(int(round(key1[0]*255)))
    i=1
    for i in range(1,512):
        y=1-float(x)
        y1=float(r)*float(x)
        z=float(y1)*float(y)
        x=z;
        #print ('ans is {0} and {1} and {2}'.format(round(y),y1,z))
        #x=float(r)*float(x)(1-float(x))
        key1.append(z)
        temp_key1.append(z)
        temp_key1[i]=int(round(key1[i]*255))
        key1[i]='{0:08b}'.format(int(round(key1[i]*255)))
        #print (key1[i])
    return key1,temp_key1
