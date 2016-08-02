'''FaceExRecog in live webcam using CNN'''

import cv2
from sklearn.externals import joblib

classes = {0: 'Disgust', 1: 'Neutral', 2: 'Surprise', 3: 'Angry',
           4: 'Fear', 5: 'Sad', 6: 'Happy', 7: 'contempt'}
cascPath = "haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

# Sizes for image to resize
wdef = 157
hdef = 157

# loading the classifier model
classifier = joblib.load('cnnpkl/cnn.pkl')  # loads cnn model

while True:
    # Capture frame by frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        color = (0, 0, 255)
        if 130 < h < 190:
            color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.equalizeHist(roi_gray)
        roi_color = frame[y:y+h, x:x+w]

        # show text in the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(w)+'X'+str(h), (x+w+5, y+h+20),
                    font, 0.5, (0, 255, 0), 2)

        # reshape to standard size
        roi_gray = cv2.resize(roi_gray, (wdef, hdef))
        roi_gray = roi_gray.reshape((-1, 1, wdef, hdef))
        # classify the results
        # result = classifier.predict(roi.reshape(1, len(roi)))
        result = classifier.predict(roi_gray)
        # print result
        cv2.putText(frame, classes[result[0]], ((x+w)/2, y+h+30),
                    font, 1, (0, 255, 0), 2)

    # display the resulting frame
    cv2.imshow('Face detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
video_capture.release()
cv2.destroyAllWindows()
