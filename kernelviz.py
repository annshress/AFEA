import cv2
import numpy as np
import matplotlib.pyplot as plt
from oct2py import octave as oc

cascPath = "haar/haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

sdgaussian = 3.14
theta = 0
spatialasp = 1

kernels = np.zeros((31,31))
for theta in range(0,181,25): # six orientation
        for lambd in [2,3,4]: # three wavelegth of sinusoidal factor
                kernels += cv2.getGaborKernel((31,31), sdgaussian, theta*3.14/180, lambd, spatialasp, psi = 0)
                
'''
for i in range(18):
	plt1 = plt.gca()
	plt1.axes.get_xaxis().set_ticks([])
	plt1.axes.get_yaxis().set_ticks([])
	plt.subplot(6,3,i+1)
	plt.imshow(kernels[i],'gray')

plt.show()
'''
image = "image.tiff"
image = cv2.imread(image,0)
image = cv2.equalizeHist(image)

face = faceCascade.detectMultiScale(
                        image,
                        scaleFactor = 1.1,
                        minNeighbors = 5,
                        minSize = (30, 30),
                        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                        )

                #Draw a rectangle around the faces
for (x, y, w, h) in face:
  roi = image[y:y+h, x:x+w]
  roi = cv2.resize(roi,(157,157))

result = []

for k in [kernels]:
  output = oc.conv2(roi,k,'same')
  result.append(output)

print result[0].shape
total = result[0]
np.savez('tempwavelet',result[0].reshape((1,total.shape[0]*total.shape[1])))
'''
total = np.zeros(result[0].shape)
for i in range(18):
  plt1 = plt.gca()
  plt1.axes.get_xaxis().set_ticks([])
  plt1.axes.get_yaxis().set_ticks([])
  plt.subplot(6,3,i+1)
  total += result[i]
  plt.imshow(result[i],'gray')
'''
result = np.load('tempwavelet.npz')['arr_0'][0]
result = result.reshape((157,157))
plt.figure(2)
plt.imshow(result,'gray')
plt.figure(3)
plt.imshow(roi,'gray')
plt.show()

