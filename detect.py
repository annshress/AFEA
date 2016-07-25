from os import listdir
from os.path import isfile,join
import numpy
import cv2
import csv

def show_pic(p,name):
        #name = 'press N for next image'
        cv2.namedWindow(name)
        cv2.imshow(name,p)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        #cv2.waitKey(1)
        
cascPath = "haar/haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

#to load images in a folder
mypath = '/home/anish/workspace/afeaproj/images'
ckpath = '/home/anish/workspace/afeaproj/CK+/cohn-kanade-images'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

# on average each face in the training dataset was found to be about 157x157
wdef = 157
hdef = 157

def main():
        # need the filenames to save wavelets for local use.
        filenames = []
        face_type = []
        roi_images = []
     
        print "loading JAFFE"
        for n in range(len(onlyfiles)):
                image = cv2.imread(join(mypath,onlyfiles[n]),0)
                #  histogram equalization
                image = cv2.equalizeHist(image)
                
                temp = onlyfiles[n][3:5]
                face = faceCascade.detectMultiScale(
                        image,
                        scaleFactor = 1.1,
                        minNeighbors = 5,
                        minSize = (30, 30),
                        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                        )

                #Draw a rectangle around the faces
                for (x, y, w, h) in face:
                        # cv2.rectangle(images[n], (x,y), (x+w, y+h), (0,140,255), 2)
                        roi = image[y:y+h, x:x+w]
                        roi_images.append(cv2.resize(roi,(wdef, hdef)))

                face_type.append(temp)
                filenames.append(onlyfiles[n][:7])
        
        print "loading CK+"
        j = 0
        with open('expressions.csv','r') as csvfile:
                # fileds = ['subject','folder','expressions']
                # j += 1
                reader = csv.DictReader(csvfile)
                for row in reader:
                        subject = row['subject']
                        folder = row['folder']
                        expression = row['expression']
                        path = join(ckpath,subject,folder)
                        images = listdir(path)
                        images.sort()
                        for n in [0,-3,-2,-1]:
                                img = cv2.imread(join(path,images[n]),0)
                                #  histogram equalization
                                img = cv2.equalizeHist(img)
                                # fails at 323
                                # try:
                                # i += 1/4.0
                                face = faceCascade.detectMultiScale(
                                                        img,
                                                        scaleFactor = 1.1,
                                                        minNeighbors = 5,
                                                        minSize = (30, 30),
                                                        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                                                        )
                                (x, y, w, h) = face[-1]
                                # cv2.rectangle(images[n], (x,y), (x+w, y+h), (0,140,255), 2)
                                roi_img = img[y:y+h, x:x+w]
                                roi_images.append(cv2.resize(roi_img,(wdef, hdef)))
                                #if len(face) == 2:
                                #       show_pic(roi_images[-1], str(h)+str(w))
                                # append first image with neutral class, else append peak expression
                                expression = 'NE' if n == 0 else row['expression']
                                face_type.append(expression)
                                filenames.append(subject+"."+expression+str(abs(n))+"."+folder[-1])
        # cv2.startWindowThread()
        # for i in range(len(roi_images)):
        #        show_pic(roi_images[i],filenames[i])

        # print len(roi_images), len(face_type), len(filenames)
        return roi_images, face_type, filenames
