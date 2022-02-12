import warnings

warnings.filterwarnings("ignore")

from main.hog import HogFeature
from main.orb import OrbFeature
from main.sift import SiftFeature

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd


def test_with_kNN(X_train, X_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    k_range = range(1, 30)
    train_score = []
    test_score = []
    best_k = 1
    best_acc = 0
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=2)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_train)
        train_score.append(metrics.accuracy_score(y_train, y_pred))

        y_pred = knn.predict(X_test)
        test_acc = metrics.accuracy_score(y_test, y_pred)
        test_score.append(test_acc)
        if test_acc >= best_acc:
            best_acc = test_acc
            best_k = k

    plt.figure(figsize=(8, 8))
    plt.plot(k_range, train_score)
    plt.plot(k_range, test_score)
    plt.xlabel('Value of K for knn')
    plt.ylabel('Accuracy')
    plt.legend(["Train Accuracy", "Test Accuracy"], loc="upper right")
    plt.show()

    classifier = KNeighborsClassifier(n_neighbors=best_k, metric="minkowski", p=2)
    classifier.fit(X_train, y_train)

    y_predict = classifier.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_predict)
    print(f'KNN Accuracy is :{accuracy}')


def test_with_decisionTree(X_train, X_test, y_train, y_test):
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_predict)
    print(f'Decision Tree Accuracy is :{accuracy}')


def test_with_logistic_regression(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    LRclassifier = LogisticRegression(random_state=0, multi_class='ovr')
    model = LRclassifier.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_predict)
    print(f'Logistic Regression Accuracy is :{accuracy}')


def test_with_svm_ovo(X_train, X_test, y_train, y_test):
    from sklearn import svm
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_predict)
    print(f'SVM  OVO Accuracy is :{accuracy}')


def test_with_svm_ovr(X_train, X_test, y_train, y_test):
    from sklearn import svm
    clf_ovr = svm.SVC(decision_function_shape='ovr')
    clf_ovr.fit(X_train, y_train)
    y_predict = clf_ovr.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_predict)
    print(f'SVM OVR Accuracy is :{accuracy}')


def show_plot(hog_features, sift_features, orb_features, y):
    # Plot
    plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
    plt.scatter(hog_features, y, label=f'y hog_features')
    plt.scatter(sift_features, y, label=f'y sift_features ')
    plt.scatter(orb_features, y, label=f'y orb_features ')
    plt.xlabel("Features Values (X)")
    plt.ylabel("Flowers Types (Y)")

    plt.title('Scatter Features')
    plt.legend()
    plt.show()


def show_sub_plots(df):
    import seaborn as sns
    sns.pairplot(df,hue='class',palette ="Paired")
    fig, axes = plt.subplots(3, 2, figsize=(12,12))
    index = 0
    for i in range(3):
        for j in range(i+1,4):
            ax1 = int(index/2)
            ax2 = index % 2
            axes[ax1][ax2].scatter(df[df.columns[i]], df[df.columns[j]], color='red')
            axes[ax1][ax2].set_xlabel(df.columns[i])
            axes[ax1][ax2].set_ylabel(df.columns[j])
            index = index + 1
    plt.figure(figsize=(16,6))
    heatmap = sns.heatmap(df.corr(), vmin=-1, vmax = 1, annot = True)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
    plt.show()


class FlowerHogSiftOrb:
    hog = HogFeature()
    hog_features = hog.getHogFeature()

    sift = SiftFeature()
    sift_features = sift.getSiftFeature()

    orb = OrbFeature()
    orb_features = orb.getOrbFeature()
    #x1 = np.ones(200)

    raw_data = np.vstack((hog_features, sift_features, orb_features)).T

  # class'larımızı ekledik
    y1 = np.full((50), 77)
    y2 = np.full((50), 73)
    y3 = np.full((50), 88)
    y4 = np.full((50), 89)
    y = np.concatenate((y1, y2, y3, y4), axis=0)

    #  multi class classifier
    # y1 = np.full((50), 0)
    # y2 = np.full((50), 1)
    # y3 = np.full((50), 0)
    # y4 = np.full((50), 0)
    # y = np.concatenate((y1, y2, y3, y4), axis=0)

    show_plot(hog_features, sift_features, orb_features, y)
    # featureları ve classı birleştirdik
    data = np.column_stack((raw_data, y))

    columnNames = ['hog','sift','orb','class']
    df = pd.DataFrame(data,columns=columnNames)
    print(df.head())
    print("----Correlation----")
    print(df.corr())
    print("--------------------")

    show_sub_plots(df)

    X = data[:, 0:3]
    y = data[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    test_with_kNN(X_train, X_test, y_train, y_test)
    test_with_decisionTree(X_train, X_test, y_train, y_test)
    test_with_logistic_regression(X_train, X_test, y_train, y_test)
    test_with_svm_ovo(X_train, X_test, y_train, y_test)
    test_with_svm_ovr(X_train, X_test, y_train, y_test)
