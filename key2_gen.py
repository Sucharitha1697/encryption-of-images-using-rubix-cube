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

def gen2(): 
    r1=input("input r between 2 and 4 : ")
    x1=input("input x between 0 and 1 : ")
    key2=[]
    temp_key2=[]
    key2.append(float(x1))
    temp_key2.append(float(x1))
    temp_key2[0]=int(round(key2[0]*255))
    key2[0]='{0:08b}'.format(int(round(key2[0]*255)))
    i=1
    for i in range(1,512):
        y2=1-float(x1)
        y3=float(r1)*float(x1)
        z1=float(y2)*float(y3)
        x1=z1;
        #print ('ans is {0} and {1} and {2}'.format(round(y),y1,z))
        #x=float(r)*float(x)(1-float(x))
        key2.append(z1)
        temp_key2.append(z1)
        temp_key2[i]=int(round(key2[i]*255))
        key2[i]='{0:08b}'.format(int(round(key2[i]*255)))
       # print (key2[i])
    return key2,temp_key2
