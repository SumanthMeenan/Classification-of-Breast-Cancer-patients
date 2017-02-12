import numpy as np

from sklearn import linear_model, svm


def main():
    
    df = np.loadtxt('breast-cancer-wisconsin.data',delimiter=',')
    
    X_train = df[:550,:10]
    Y_train = df[:550,10]
    
    X_test = df[550:,:10]
    Y_test = df[550:,10]
    
    
    logistic_regression = linear_model.LogisticRegression()
    linear_svc = svm.LinearSVC()
    logistic_regression = linear_model.Perceptron()    
    #print Y_train
    #print Y_test

    model1 = logistic_regression.fit(X_train,Y_train)
    model2 = linear_svc.fit(X_train,Y_train)
    Y_predicted1 = model1.predict(X_test)
    Y_predicted2 = model2.predict(X_test)
    
    print Y_predicted1
    print Y_predicted2
    print model1.score(X_test,Y_test)
    print model2.score(X_test,Y_test)
    
if __name__=="__main__":
    main()
