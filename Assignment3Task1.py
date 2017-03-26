from sklearn.svm import LinearSVC

from preamble import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


#Standard OML get data stuff
oml.config.apikey = '366585f9ec2435e0f2543d5175e1bac5'
mnist_data = oml.datasets.get_dataset(554) # Download MNIST data
# Get the predictors X and the labels y
X, y = mnist_data.get_data(target=mnist_data.default_target_attribute)
X = X[:500]
y = y[:500]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30, train_size=0.8)

#Use the following values of $k=45,90,400$ in your experiments.
#K is the number of dimensions in the created subspace
k=45

#D is the number of dimensions in the original dataset
d=784

rMatrix = [[0 for aaaaa in range(k)] for bbbbb in range(d)]
for i in range(0,d):
    for j in range(0,k):
        coinFlip = np.random.randint(0,2)
        if (coinFlip == 0):
            rMatrix[i][j] = 1/d
        else:
            rMatrix[i][j] = -(1/d)


fP= np.dot(X,rMatrix)

p = np.amax(X)
q = np.amax(y)

randProjX = abs(p*np.array(fP) - q*np.array(fP))/ abs(p - q)

plt.hist(randProjX)
plt.show()

print(np.amax(randProjX))
#
# ###Part 2
# kValues = [45,90,400]
#
# originalAccuracies = []
# transformedAccuracies = []
#
# # loop over the projection sizes
#
#
# # train a classifier on the initial values
# originalModel = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
# originalModel.fit(X_train, y_train)
# # evaluate the model and update the list of accuracies
# originalAccuracies.append(metrics.accuracy_score(originalModel.predict(X_test), y_test))
#
#
# # create the random projection
# X_trainRand, X_testRand, y_trainRand, y_testRand = train_test_split(randProjX, y, random_state=30, train_size=0.8)
# # train a classifier on the sparse random projection
# rpModel = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
# rpModel.fit(X_trainRand, y_trainRand)
# # evaluate the model and update the list of accuracies
# transformedAccuracies.append(metrics.accuracy_score(rpModel.predict(X_testRand), y_testRand))
#
# print("Original accuracy sores are:")
# print(originalAccuracies)
# print("Accuracy scores using random projection are:")
# print(transformedAccuracies)
#
#
# #
# # # #Check without random projection
# # # nn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
# # # nn.fit(X_train,y_train)
# # # print("Training set score: {:.3f}".format(nn.score(X_train, y_train)))
# # # print("Test set score: {:.3f}".format(nn.score(X_test, y_test)))
