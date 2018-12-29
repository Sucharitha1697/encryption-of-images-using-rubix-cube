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
import correlation_coeff as fn1
import npcr as fn2
import uaci as fn3
import key1_gen as fn4
import key2_gen as fn5

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

class Cube:
    matrix = []

    def __init__(self, *args):
        if args.__len__() == 2:
            rows = args[0]
            columns = args[1]
            self.matrix = [[0 for x in range(columns)] for x in range(rows)]
            for i in range(rows):
                for j in range(columns):
                    self.matrix[i][j] = random.randint(0, 255)
        elif args.__len__() == 1:
            if type(args[0]) is list:
                self.matrix = copy.deepcopy(args[0])
            else:
                self.matrix = copy.deepcopy(args[0].matrix)
        else:
            sys.exit('Too few arguments given to Cube constructor.')

    def print_cube(self):
        for i in range(self.matrix.__len__()):
            for j in range(self.matrix[0].__len__()):
                if self.matrix[i][j] < 100:
                    print('0', end="")
                if self.matrix[i][j] < 10:
                    print('0', end="")

                print(str(self.matrix[i][j]) + " ", end="")
            print()

    def rotate_clockwise(self):
        self.matrix = np.rot90(self.matrix, -1)

    def rotate_anti_clockwise(self):
        self.matrix = np.rot90(self.matrix, 1)

    def rotate_column(self, column, times):
        self.matrix = np.transpose(self.matrix)
        np.roll(self.matrix[column], times)
        self.matrix = np.transpose(self.matrix)

    def rotate_row(self, row, times):
        self.matrix[row] = np.roll(self.matrix[row], times)


class Move(Enum):
    rotation_row = 1
    rotation_column = 2
    rotation_clockwise = 3
    rotation_anti_clockwise = 4


class Movement:
    type = None
    times = 0
    row_or_column = 0

    def __init__(self, rows, columns):
        self.type = random.choice(list(Move))
        if self.type == Move.rotation_column:
            self.times = random.randint(1, columns-1)
            self.row_or_column = random.randint(0, columns-1)
        elif self.type == Move.rotation_row:
                self.times = random.randint(1, rows-1)
                self.row_or_column = random.randint(0, rows-1)

def encrypt(cube, moves):
    for rotation in moves:
        if rotation.type == Move.rotation_row:
            cube.rotate_row(rotation.row_or_column, rotation.times)
        elif rotation.type == Move.rotation_column:
            cube.rotate_column(rotation.row_or_column, rotation.times)
        elif rotation.type == Move.rotation_clockwise:
            cube.rotate_clockwise()
        elif rotation.type == Move.rotation_anti_clockwise:
            cube.rotate_anti_clockwise()


def decrypt(cube, moves):
    for rotation in reversed(moves):
        if rotation.type == Move.rotation_row:
            cube.rotate_row(rotation.row_or_column, cube.matrix.__len__() - rotation.times)
        elif rotation.type == Move.rotation_column:
            cube.rotate_column(rotation.row_or_column, cube.matrix[0].__len__() - rotation.times)
        elif rotation.type == Move.rotation_clockwise:
            cube.rotate_anti_clockwise()
        elif rotation.type == Move.rotation_anti_clockwise:
            cube.rotate_clockwise()

#logistic map

key1=[]
temp_key1=[]
key1,temp_key1=fn4.gen1()

#logistic map2

key2=[]
temp_key2=[]
key2,temp_key2=fn5.gen2()


j=0
k=7
fin=""
temp1=0
for j in range(0,8):
    temp=int(key1[256][j])^int(key2[256][j])
    temp1=int(temp1)+(2**k)*int(temp)
    k=k-1
    fin+=str(temp)
ans=int(temp1)%5+10
#print(ans.type())
print('Final no of iterations : ', ans)
#print(fin)
#print(temp1)

img = misc.imread("lena_gray.jpg",flatten=True)[0:512, 0:512]
#img=rgb2gray(img)
img_size=img.shape
#print(img_size)
#print(img)


temp=img
for kp in range(0,ans):
 increment =256

 moves = []
 for i in range(512):
    moves.append(Movement(increment, increment))

 iterations =range(0, 512, increment)
 #for t in iterations:
  #print(t)

 encrypted_image = np.array([x[:] for x in [[0] * 512] * 512])
 decrypted_image = np.array([x[:] for x in [[0] * 512] * 512])

 for X in iterations:
     for Y in iterations:
         cube = Cube(list(temp[X:X+ increment, Y:Y+ increment]))
         encrypt(cube, moves)
         encrypted_image[X:X + increment, Y:Y + increment] = cube.matrix
         decrypt(cube, moves)
         decrypted_image[X:X + increment, Y:Y + increment] = cube.matrix

 
 encrypted_image = encrypted_image.astype(np.uint8)
 decrypted_image = decrypted_image.astype(np.uint8)

 for i in range(512):
     for j in range(512):
         encrypted_image[i][j]=int(encrypted_image[i][j])^int(temp_key1[j])
 for i in range(512):
     for j in range(512):
         encrypted_image[j][i]=int(encrypted_image[j][i])^int(temp_key2[j])
 temp=encrypted_image
 """plt.imshow(encrypted_image,cmap='gray')
 #print(encrypted)
 plt.title("Encrypted image for iteration no : %d" %kp)
 plt.show()"""
 
decrypted_image=img     
img = img.astype(np.uint8)

# Assert that the dimensions of original and decrypted are equal
assert img.__len__() == decrypted_image.__len__()
assert img.__len__() == decrypted_image[0].__len__()

# Assert that each pixel in original and decrypted are the same
dimensions = [img.__len__(), img[0].__len__()]
for i in range(dimensions[0]):
 for j in range(dimensions[1]):
    assert img[i][j] == decrypted_image[i][j]

print (encrypted_image)
print ('The npcr value is : ', fn2.npcr(img,encrypted_image))
print ('The uaci value is : ', fn3.uaci(img,encrypted_image))
print ('The corr value of original img is : ', fn1.corr(img))
print ('The corr value is encrypted img is : ', fn1.corr(encrypted_image))
#print(entropy(img))
#print(entropy(encrypted_image))

# Plot of all three figures showing the original, encrypted, and decrypted

plt.imshow(img, cmap='gray')
plt.title("Original")
plt.show()


plt.imshow( encrypted_image,cmap='gray')
#print(encrypted)
plt.title("Final Encrypted Image")
plt.show()


plt.imshow( decrypted_image, cmap='gray')
#print(decrypted)
plt.title("Final Decrypted Image")
plt.show()
    
     



 


 
