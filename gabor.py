import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from oct2py import octave as oc
from math import pi

import detect
newpath = r'kernels/'

def filter(u=5,v=8):

  try: # read from a file
    # raise IOError
    if not os.path.exists(newpath):
      os.makedirs(newpath)
      raise IOError
    else:
      kernels = np.load('kernels/k.npy')
  except IOError:
    print "ioerror, creating new kernel"
    ksize = 31
    kernels = np.zeros(shape = (ksize,ksize))
    sigma = pi # sd of gaussian envelope or bandwidth
    gamma = 1 # spatial aspect ratio
    #fmax = 0.25

    #for i in range(1,u+1):
    # cv2.startWindowThread()
    # cv2.namedWindow("image")
    for theta in range(0,181,25): # six orientation
      for lambd in [7,8,9]: # three wavelegth of sinusoidal factor
        #lambd = fmax/pow(sqrt(2),i-1)
        kernel = cv2.getGaborKernel((ksize,ksize), sigma, theta*pi/180, lambd , gamma, psi = 0)
        kernel /= 1.5*kernel.sum()
        # rule of distributivity of convolution
        kernels += kernel
    # cv2.destroyAllWindows() 
    # will return 6x3 = 1 kernels
    np.save('kernels/k',kernels)

  return kernels

def PCA(wavelets,features):
  keys = wavelets.keys()
  # they have specific order, probably: ['DI', 'NE', 'SU', 'AN', 'FE', 'SA', 'HA']
  temp_wav = [] # 1-D wavelets of each wavelets
  counts = [] # counts of each face_types
  for key in keys:
    wave_key = wavelets[key]
    counts.append(len(wave_key))
    for each_wav in wave_key: 
      new_wav = cv2.resize(each_wav,(0,0),fx = 0.675, fy = 0.675)
      temp_wav.append(new_wav.reshape(new_wav.size,1)) # 2D faces to 1D faces
      
  u,ss,v = np.linalg.svd([temp_wav])
  s = np.diag(ss)

  u = u[1:features]
  face = s*v.T
  
  return u,s,v

def convolute(img,kernel):
  output = oc.conv2(img,kernel,'same')
  # output = cv2.filter2D(img, cv2.CV_64F, kernel)
  #plt.imsave(str(i)+str(j),output, cmap = plt.cm.gray)
  return output/18

def main():
  ker = filter()

  roi_images, face_type, file_names = detect.main() # images and their respective classes

  order = ['DI', 'NE', 'SU', 'AN', 'FE', 'SA', 'HA', 'CO']
  # order of arrangement
  
  wavelets = {'HA': [],'SA': [],'SU': [],'AN': [],'DI': [],'FE': [],'NE':[],'CO':[]}
  # array of the wavelet images OF SPECIFIC expressions
  # CLASSIFYING GUYS: NEEDS ABOVE WAVELETS DICTIONARY
  i = 0
  print "saving wavelets"
  for img in roi_images:
    total = convolute(img,ker)
    # wavelets[face_type[i]].append(total.reshape((1,total.shape[0]*total.shape[1])))
    np.savez('wavelets(roi-789-ck)/'+str(file_names[i][:10]),total.reshape((1,total.shape[0]*total.shape[1])))
    i+=1
  print i
  
  print "wavelets loaded"
  '''
  classes = []
  waves = []
  for key in order:
    for i in range(len(wavelets[key])):
        classes.append(key)
    for each in wavelets[key]:
      waves.append(each[0])
  # print face_type
  # return classes, waves
  #u,s,v = PCA(wavelets,3) # will use PCA when we need it.
  #np.savez('wav.npz',wavelets)
  #200 images goes to 40 MB.
  '''
