import random

# our classes
def shuffle(classes):
    alist = classes
    blist = range(len(alist))
    aSet = set(alist)
    aDict = {}

    for item in aSet:
    	aDict[item] = alist.count(item)
    #print aDict

    # we will use 70% items from each class
    items = 0.7

    train_index = []
    test_index = []

    for key in aSet:
        start = alist.index(key)
        end = start + aDict[key]
        print key, " starts :", start, " ends: ",end
        temp = range(start,end)
        random.shuffle(temp)
        limit = int(0.7*len(temp))
        train_index += temp[:limit]
        test_index += temp[limit:]

    # train_index are our index_shuf
    random.shuffle(train_index)
    #print train_index
    return train_index,test_index
    '''
    train_x = []
    train_y = []
    for i in train_index:
        train_x.append(alist[i])
        train_y.append(blist[i])
    print "/n/n/nstratify.py"
    print "trainX = ",train_x
    print "trainY = ",train_y
    '''