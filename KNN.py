
import numpy as np
import cv2


def to_centre(mat):
    x = 0
    y = 0
    cnt = 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if (mat[i][j] != 0):
                x += i
                y += j
                cnt += 1
    #if(cnt==0):
    offset_x = int(x / cnt) - int(mat.shape[0] / 2)
    offset_y = int(y / cnt) - int(mat.shape[1] / 2)
    ret = np.empty((mat.shape[0], mat.shape[1]), np.uint8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            x = i + offset_x
            y = j + offset_y
            if(x >= 0 and x < mat.shape[0] and y >= 0 and y < mat.shape[1]):
                ret[i][j] = mat[x][y]
            else:
                ret[i][j] = 0
    return ret
 
def KNN(inX, dataSet, labels, k, iNormNum):
    subtractMat = np.ones([dataSet.shape[0], 1])*np.array(inX).reshape([1, inX.shape[1]]) - dataSet
    distances = ((subtractMat**iNormNum).sum(axis=1))**(1.0/iNormNum)
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda s: s[1], reverse=True)
    return sortedClassCount[0][0]
#    return labels[sortedClassCount[0][0]]
     
if __name__ == '__main__':

    data_num1 = 60000 #The number of figures
    data_num2 = 10000
    fig_w = 45       #width of each figure

    train_features  = np.fromfile("mnist_train_data",dtype=np.uint8)
    train_labels = np.fromfile("mnist_train_label",dtype=np.uint8)
    test_features= np.fromfile("mnist_test_data",dtype=np.uint8)
    test_labels = np.fromfile("mnist_test_label",dtype=np.uint8)

    #print(train_features)

    train_features = train_features.reshape(data_num1,fig_w,fig_w)
    test_features = test_features.reshape(data_num2,fig_w,fig_w)

    #data_num1 = 1000 #The number of figures
    #data_num2 = 200
    #train_features = train_features[:data_num1]
    #test_features = test_features[:data_num2]
    #train_labels=train_labels[:data_num1]
    #test_labels=test_labels[:data_num2]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    for i in range(train_features.shape[0]):
    #for i in range(100):
        #print(train_features[i])
        cv2.threshold(train_features[i],128,255,cv2.THRESH_BINARY,train_features[i])
        #train_features[i] = cv2.morphologyEx(train_features[i], cv2.MORPH_OPEN, kernel)
        train_features[i] = to_centre(train_features[i])
        train_features[i] = cv2.morphologyEx(train_features[i], cv2.MORPH_OPEN, kernel)
    for i in range(test_features.shape[0]):
    #for i in range(100):
        #print(train_features[i])
        cv2.threshold(test_features[i],128,255,cv2.THRESH_BINARY,test_features[i])
        #test_features[i] = cv2.morphologyEx(test_features[i], cv2.MORPH_OPEN, kernel)
        test_features[i] = to_centre(test_features[i])
        test_features[i] = cv2.morphologyEx(test_features[i], cv2.MORPH_OPEN, kernel)

    train_features = train_features.reshape(data_num1,fig_w*fig_w)
    test_features = test_features.reshape(data_num2,fig_w*fig_w)
    #testDataSet = testDataSet[:100]

    #data_num2=100

    iErrorNum      = 0

    for iTestInd in range(data_num2):
        KNNResult = KNN(test_features[iTestInd].reshape([1,45*45]), train_features, train_labels, 5, 2)
        if (KNNResult != test_labels[iTestInd]): iErrorNum += 1.0
        print("process:%d/%d_totalErrorNum:%d predict_label: %d, real_label: %d" % (iTestInd, data_num2, iErrorNum, KNNResult, test_labels[iTestInd]))
    print("\nthe total number of errors is: %d" % iErrorNum)
    print("\nthe total error rate is: %f" % (iErrorNum/float(data_num2)))
    print("\naccuracy : %f" % (1-(iErrorNum/float(data_num2))))
     
