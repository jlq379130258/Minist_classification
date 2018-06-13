
import numpy as np
import cv2

 
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

    train_features = np.fromfile("mnist_train_data",dtype=np.uint8)
    train_labels = np.fromfile("mnist_train_label",dtype=np.uint8)
    test_features = np.fromfile("mnist_test_data",dtype=np.uint8)
    test_labels = np.fromfile("mnist_test_label",dtype=np.uint8)


    train_features = train_features.reshape(60000,45,45)
    train_features = train_features.astype(np.float32)
    test_features = test_features.reshape(10000,45,45)
    test_features = test_features.astype(np.float32)
    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)

    train_features=train_features.flatten()
    train_features=train_features.reshape(60000,45*45)
    test_features=test_features.flatten()
    test_features=test_features.reshape(10000,45*45)

    iErrorNum      = 0

    for iTestInd in range(10000):
        KNNResult = KNN(test_features[iTestInd].reshape([1,45*45]), train_features, train_labels, 5, 2)
        if (KNNResult != test_labels[iTestInd]): iErrorNum += 1.0
        print("process:%d/%d_totalErrorNum:%d predict_label: %d, real_label: %d" % (iTestInd, 10000, iErrorNum, KNNResult, test_labels[iTestInd]))
    print("\nthe total number of errors is: %d" % iErrorNum)
    print("\nthe total error rate is: %f" % (iErrorNum/float(10000)))
    print("\naccuracy : %f" % (1-(iErrorNum/float(10000))))
     
