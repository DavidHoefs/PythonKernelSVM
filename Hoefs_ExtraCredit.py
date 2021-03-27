# AUTHOR: David Hoefs R#11481012
# Extra Credit Assignment
# Pattern Recognition Spring 2021
# Environment: Visual Studio Code with Python 3.9.0
# =============================== References ===============================
# https://cvxopt.org/examples/tutorial/qp.html - Documentation on cvxopt python package
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html - matplotlib contour plot documentation


import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
from numpy.core.defchararray import array
import pandas as pd
from matplotlib import pyplot as pl
import seaborn as sns
sns.set()
import pylab as pl



def gaussianKernelFunction(x, y, sigma):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

# Read data from excel dataset and split into X and y 
# returns X,y containing data
def getData():
        df = pd.read_excel("C:\\Users\\hoefs\\PatternRecognition\\ExtraCreditv2\\ExtraCreditDataset.xlsx",sheet_name='Sheet1' ,engine='openpyxl',header = None)
        numArray = df.to_numpy()
        X = numArray[0:,0:2]
        y = numArray[:, 2]
        return X,y

# Creates a Kernel Matrix of size (100,100) containing values based on the results of X data being passed into gaussian function
# Returns the filled matrix
def generateKernelMatrix(X,numOfSamples,sigma):
    kernelMatrix = np.zeros((numOfSamples,numOfSamples))
    for i in range(numOfSamples):
        for j in range(numOfSamples):
            kernelMatrix[i,j] = gaussianKernelFunction(X[i],X[j],sigma)
    return kernelMatrix

# Prepares data for use with cvxopt.solvers.qp function
# Returns P,q,G,h,A,b
def getParamsForCvxoptFunction(X,y,C,numOfSamples):
    # generate different matrix params needed by cvxopt from X,y,C and the number of samples
    y = y.T
    Y = y.reshape(-1,1) * 1.
    P = np.outer(y,y) * kernelMatrix
    q = -1*np.ones(shape=(numOfSamples,1))
    G = np.append(-1*np.eye(numOfSamples),np.eye(numOfSamples), axis=0)
    h = np.append(np.zeros(shape=(numOfSamples,1)),C*np.ones(shape=(numOfSamples,1)),axis=0)
    A = Y.T
    b = np.zeros(shape=(1,1))

    # put in form for cvxopt to accept
    P = cvxopt.matrix(P.astype(np.float64))
    q = cvxopt.matrix(q.astype(np.float64))
    G = cvxopt.matrix(G.astype(np.float64))
    h = cvxopt.matrix(h.astype(np.float64))
    A = cvxopt.matrix(A.astype(np.float64))
    b = cvxopt.matrix(b.astype(np.float64))

    # return params
    return P,q,G,h,A,b

# Finds the bias
def findB(alphas,lambdas,supportVectors,supportVectorsY,kernelMatrix):
    index =np.arange(len(lambdas))[supportVectors]
    b = 0.0
    for i in range(len(alphas)):
        b += supportVectorsY[i]
        b -= np.sum(alphas * supportVectorsY * kernelMatrix[index[i],supportVectors])
    b /= len(alphas)
    return b

# Plots the feature vectors
def plotFeatureVectors(X,y):
    # get points in class 1
    positiveX = X[y==1]
    # get points in class 2
    negativeX = X[y == -1]
    # plot them
    pl.plot(positiveX[:,0],positiveX[:,1],'bo',label='Class 1')
    pl.plot(negativeX[:,0],negativeX[:,1],'ro',label='Class 2')

# Plots the support vectors
def plotSupportVectors(supportVectors):
    pl.scatter(supportVectors[:,0],supportVectors[:,1] ,s=125,c='g',label='Support Vectors')
    pl.xlabel('x1')
    pl.ylabel('x2')

# Generates Z to be used with python's contour() plot function   
def generateZ(alphas, X,supportVectorY,supportVectors,b):
    y_predict = np.zeros(len(X))
    for i in range(len(X)):
        s = 0
        for j in range(len(alphas)):
            s += alphas[j] * supportVectorY[j] *gaussianKernelFunction(X[i], supportVectors[j],1.75)
        y_predict[i] = s
    return y_predict + b

# Finds and plots the misclassifications
def findMisclassifications(X,alphas,supportVectorsY,supportVectors,b):
    y_predict = np.zeros(len(X))
    for i in range(len(X)):
        s = 0
        for j in range(len(alphas)):
            s += alphas[j] * supportVectorsY[j] *gaussianKernelFunction(X[i], supportVectors[j],1.75)
        y_predict[i] = s + b
    
    c1y = y_predict[0:60]
    c2y= y_predict[61:100]
    c1Miss = np.argwhere(c1y < 0).flatten()
    c2Miss = (np.argwhere(c2y > 0) + 61).flatten()
    
    miss1Nums = X[c1Miss]
    miss2Nums = X[c2Miss]
    
    pl.plot(miss1Nums[:,0],miss1Nums[:,1],'md',label='Misclassified', ms=15,fillstyle='none')
    pl.plot(miss2Nums[:,0],miss2Nums[:,1],'md',ms=15,fillstyle='none')

    return len(c1Miss) + len(c2Miss)
# plot the decision boundry, and 2 margin hyperplanes
def plotBoundaries(X,alphas,supportVectorsY,supportVectors,b):

    X1, X2 = np.meshgrid(np.linspace(-3,10,100), np.linspace(-3,10,100)) 
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = generateZ(alphas,X,supportVectorsY,supportVectors,b)
    Z = Z.reshape(X1.shape)
    Z_lower = Z - 1.0
    Z_upper = Z + 1.0
    C1=pl.contour(X1, X2, Z,[0], colors='k', linewidths=1, origin='lower')
    C2=pl.contour(X1, X2, Z_upper, [0], colors='r', linewidths=1, origin='lower',linestyles='dashed')
    C3=pl.contour(X1, X2, Z_lower, [0], colors='b', linewidths=1, origin='lower',linestyles='dashed')
    lines = [ C1.collections[0], C2.collections[0], C3.collections[0]]
    labels = ['decision boundry','margin hyperplane','margin hyperplane']
    pl.xlabel('x1')
    pl.ylabel('x2')
    pl.yticks(np.arange(-3, 10, step=1))
    pl.xticks(np.arange(-3, 10, step=1))
    pl.axis("tight")
    return lines,labels

X,y = getData()
numOfSamples,numOfFeatures = X.shape
sigma = 1.75

#=================================== For C = 10 ===================================
# generate kernel matrix 
kernelMatrix = generateKernelMatrix(X,numOfSamples,sigma)
# set value of C
C = 10
# get params to pass into cvxopt.qp function
P,q,G,h,A,b = getParamsForCvxoptFunction(X,y,C,numOfSamples)
# call cvxopt.solvers.qp and pass in params from above
solverSolution = cvxopt.solvers.qp(P,q,G,h,A,b)
# get lambas from solution
lambdas = np.ravel(solverSolution['x'])
# get the support vectors , returns boolean array that will be used to index into support vectors to get actual values
supportVectorsBool = ((lambdas > 1e-3) * (lambdas < C)).flatten() 
# get the support vector lambdas
alphas = lambdas[supportVectorsBool]
# get the support vectors from the feature vectors, returns array with actual X values that are support vectors
supportVectors = X[supportVectorsBool]
# gets the y values corresponding to the support vectors
supportVectorsY = y[supportVectorsBool]
# find the bias b
b = findB(alphas,lambdas,supportVectorsBool,supportVectorsY,kernelMatrix)



pl.figure(1,figsize=(10,5))
# pl.figure(figsize=(25,25))

print('b = ',b)
print('Support Vectors: ', len(supportVectors))
plotFeatureVectors(X,y)
plotSupportVectors(supportVectors)
lines,labels = plotBoundaries(X,alphas,supportVectorsY,supportVectors,b)
numMisclassified = findMisclassifications(X,alphas,supportVectorsY,supportVectors,b)
pl.title('Gaussian Kernel: \n Support Vectors = ' + str(len(supportVectors)) + '\nC=' + str(C) + '\nMisclassified= ' + str(numMisclassified))
contourLegends = pl.legend(lines,labels)
pl.legend(loc='lower right')
pl.gca().add_artist(contourLegends)
# pl.show()

#=================================== For C = 100 ===================================

kernelMatrix = generateKernelMatrix(X,numOfSamples,sigma)
# set value of C
C = 100
# get params to pass into cvxopt.qp function
P,q,G,h,A,b = getParamsForCvxoptFunction(X,y,C,numOfSamples)

# call cvxopt.solvers.qp and pass in params from above
solverSolution = cvxopt.solvers.qp(P,q,G,h,A,b)
# get lambas from solution
lambdas = np.ravel(solverSolution['x'])
# get the support vectors , returns boolean array that will be used to index into support vectors to get actual values
supportVectorsBool = ((lambdas > 1e-3) * (lambdas < C)).flatten() 
# get the support vector lambdas
alphas = lambdas[supportVectorsBool]
# get the support vectors from the feature vectors, returns array with actual X values that are support vectors
supportVectors = X[supportVectorsBool]
# gets the y values corresponding to the support vectors
supportVectorsY = y[supportVectorsBool]
# find the bias  b
b = findB(alphas,lambdas,supportVectorsBool,supportVectorsY,kernelMatrix)
pl.figure(2,figsize=(10,5))
print('b = ',b)
print('Support Vectors: ', len(supportVectors))
plotFeatureVectors(X,y)
plotSupportVectors(supportVectors)
lines,labels = plotBoundaries(X,alphas,supportVectorsY,supportVectors,b)
numMisclassified = findMisclassifications(X,alphas,supportVectorsY,supportVectors,b)
pl.title('Gaussian Kernel: \n Support Vectors = ' + str(len(supportVectors)) + '\nC=' + str(C)+ '\nMisclassified= ' + str(numMisclassified))
contourLegends = pl.legend(lines,labels)
pl.legend(loc='lower right')
pl.gca().add_artist(contourLegends)
pl.show()




