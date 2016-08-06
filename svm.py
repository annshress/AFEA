import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn import preprocessing
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
import cv2
### CUSTOM module imports
#import gabor
from stratify import shuffle

d = {'DI':1, 'NE':2, 'SU':3, 'AN':4, 'FE':5, 'SA':6, 'HA':7, 'CO':8}
d_inv = dict((v,k) for k,v in d.iteritems())

def show_pic(p):
        name = 'image'
        cv2.namedWindow(name)
        cv2.imshow(name,p)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

def main():
    if raw_input("did you check the folder name to extract wavelets from, and create particular folder to save the classifier? : (q for quit)\n") == 'q': return
    folder = '246'
    confusion =  {'DI':[0,0,0,0,0,0,0,0], 'NE':[0,0,0,0,0,0,0,0], 'SU':[0,0,0,0,0,0,0,0], 'AN':[0,0,0,0,0,0,0,0], 'FE':[0,0,0,0,0,0,0,0], 'SA':[0,0,0,0,0,0,0,0], 'HA':[0,0,0,0,0,0,0,0], 'CO':[0,0,0,0,0,0,0,0]}
    #classes, waves = gabor.main()
    mypath = "./wavelets(roi-"+folder+"-new)/"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]
    onlyfiles.sort(key = lambda string : string.split(".")[1])
    waves = [] # np.empty(len(onlyfiles), dtype=object)
    classes = []
    for n in xrange(len(onlyfiles)):
        path = mypath + onlyfiles[n]
        temp = np.load(path)
        waves.append(temp['arr_0'][0])
        # loaded wave(lets) have a dimension of 1 x 157squared.
        #show_pic(waves[n].reshape((157,157)))
        #print waves[n], type(waves[n])
        classes.append(onlyfiles[n].split(".")[1][:2])
        # classes are extracted right off the file names

    # X = preprocessing.scale(waves, axis=0, with_mean=True, with_std=True
    print "loaded wavelets"
    X = preprocessing.normalize(waves, norm="l1", axis=0, copy=True) # norm="l2", axis=1, copy=True default values
    # axis = 1, normalize each sample independently. = 0, normalize indivisual features
    # return
    print "normalizing wavelets - done"

    y = [d[i] for i in classes]
    #print y
    #print waves

    train_index, test_index = shuffle(classes)

    X_train,y_train,X_test,y_test = [],[],[],[]
    for i in train_index:
        X_train.append(X[i])
        y_train.append(y[i])
    for i in test_index:
        X_test.append(X[i])
        y_test.append(y[i])
    
    print "train and test datasets created"
    
    #return X_train, y_train, X_test, y_test
    classifier = OneVsRestClassifier(svm.SVC(kernel="rbf", gamma=0.7, C=1.0))
    
    classifier.fit(X_train, y_train)
    
    joblib.dump(classifier, 'clf'+folder+'/clf.pkl')
    
    print "Expected values of test data"
    print y_test
    
    count = 0
    j=0
    print "Obtained"
    for i in X_test:
        val = classifier.predict(i.reshape(1,-1))[0]
        if (val ==  y_test[j]):
            count = count +1
            confusion[d_inv[y_test[j]]][val-1] += 1
        else:
            confusion[d_inv[y_test[j]]][val-1] += 1
        j = j+1
        print val,",",
    
    print
    print "  DI','NE','SU','AN','FE','SA','HA','C0'"
    for each in confusion.iteritems():
        print each
    print "Accuracy"
    print float(count)/len(y_test)
    
main()

