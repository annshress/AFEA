import cv2
import sys
from sklearn.externals import joblib
from sklearn import preprocessing
import gabor

classes = {1:'Disgust', 2:'Neutral', 3:'Suprise', 4:'Angry', 5:'Fear', 6:'Sad', 7:'Happy'}
cascPath = ["haar/haarcascade_frontalface_default.xml","haar/haarcascade_eye.xml","haar/haarcascade_mcs_nose.xml","haar/haarcascade_mcs_mouth.xml"]
faceCascade = cv2.CascadeClassifier(cascPath[0])

# eyeCascade = cv2.CascadeClassifier(cascPath[1])
# noseCascade = cv2.CascadeClassifier(cascPath[2])
# mouthCascade = cv2.CascadeClassifier(cascPath[3])

video_capture = cv2.VideoCapture(0)
# from detect.py
wdef = 157
hdef = 157
classifier = joblib.load('clf789/clf.pkl')
kernels = gabor.filter()

color = (0,255,0)
x = 0
y = 0

while True:
	#Capture frame by frame
	ret, frame = video_capture.read()
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor = 1.1,
		minNeighbors = 5,
		minSize = (30, 30),
		flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	)

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		# if 140 < h < 180:
		# 	color = (0,255,0)
		cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)

		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]

		#show text in the frame
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame, str(w)+'X'+str(h), (x+w+5,y+h+20), font, 0.5, (0,255,0), 2) 
		
		# classify the roi_face
		# if color == (0,255,0):
		# reshape to standard size
		roi_gray = cv2.resize(roi_gray,(wdef, hdef))
		# call for the features
		roi = gabor.convolute(roi_gray, kernels)
		# normalize the data
		roi_norm = preprocessing.normalize(roi, norm="l2", axis=1, copy=True)
		# classify the results
		result = classifier.predict(roi_norm.reshape((1,wdef*hdef)))[0]
		cv2.putText(frame, classes[result], (x+w/2+5,y+h/2+30), font, 0.5, (0,255,0), 2) 
		# showing the region of face
		# cv2.imshow('face',roi_gray)

	#display the resulting frame
	cv2.imshow('Face detection', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
# Release the capture
video_capture.release()
cv2.destroyAllWindows()
