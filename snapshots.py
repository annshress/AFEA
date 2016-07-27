import cv2
import sys

expressions = {1:'Disgust', 2:'Neutral', 3:'Suprise', 4:'Angry', 5:'Fear', 6:'Sad', 7:'Happy',8:'Contempt'}
classes = {1:'DI', 2:'NE', 3:'SU', 4:'AN', 5:'FE', 6:'SA', 7:'HA',8:'CO'}
cascPath = ["haar/haarcascade_frontalface_alt.xml","haar/haarcascade_eye.xml","haar/haarcascade_mcs_nose.xml","haar/haarcascade_mcs_mouth.xml"]
faceCascade = cv2.CascadeClassifier(cascPath[0])
#kernel = gabor.filter(2,4,6)
#classifier = joblib.load('clf246/clf.pkl')
font = cv2.FONT_HERSHEY_SIMPLEX

video_capture = cv2.VideoCapture(0)

# from detect.py
wdef = 157
hdef = 157

color = (0,255,0)
x = 0
y = 0

subject = raw_input("Enter your first name (first four letters) : ")
print "MAINTAIN APPROPRIATE DISTANCE :: "
print "press ' q ' to quit"
print "enter the number for particular expression you will be showing to the camera."
for k in iter(classes):
    print k," : ",classes[k] 

while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    
    roi_gray = roi = gray
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # if 140 < h < 180:
        #   color = (0,255,0)
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        '''    
        roi_gray = cv2.equalizeHist(roi_gray)
        roi_gray = cv2.resize(roi_gray,(wdef,hdef))
        roi = gabor.convolute(roi_gray, kernel)
        #cv2.imshow('GrayScale', roi)
        # normalize the data
        #roi_norm = preprocessing.normalize(roi, norm="l2", axis=1, copy=True)
        #show text in the frame
        #print roi_norm.shape
        '''
        cv2.putText(frame, str(w)+'X'+str(h), (x+w+5,y+h+20), font, 0.5, (0,255,0), 2)
    
    #display the resulting frame
    cv2.imshow('Face detection', frame)
    #print classes[cv2.waitKey(1)]
    key = cv2.waitKey(1)
    if key & 0xFF in range(49,57):
        print expressions[key-48]
        cv2.imwrite("camera/"+subject+"."+classes[key-48]+".png",gray)
        #cv2.imwrite("camera/"+subject+"."+classes[key-48]+"roi.png",roi_gray)
        #cv2.imwrite("camera/"+subject+"."+classes[key-48]+"feat.png",roi)
    elif key & 0xFF == [ord('q')]: break

#release the capture
video_capture.release()
cv2.destroyAllWindows()

