from sklearn.svm import LinearSVC

from preamble import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


#Standard OML get data stuff
oml.config.apikey = '366585f9ec2435e0f2543d5175e1bac5'
mnist_data = oml.datasets.get_dataset(554) # Download MNIST data
# Get the predictors X and the labels y
X, y = mnist_data.get_data(target=mnist_data.default_target_attribute)
#Take only first 500 elements
X = X[:500]
y = y[:500]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30, train_size=0.8)

#Use the following values of $k=45,90,400$ in your experiments.
#K is the number of dimensions in the created subspace
k=45

#D is the number of dimensions in the original dataset
d=784

#Create the R matrix
rMatrix = [[0 for aaaaa in range(k)] for bbbbb in range(d)]
for i in range(0,d):
    for j in range(0,k):
        coinFlip = np.random.randint(0,2)
        if (coinFlip == 0):
            rMatrix[i][j] = 1/d
        else:
            rMatrix[i][j] = -(1/d)

storepart1 = []
storepart2 = []

storeEDistances = []
rTransp = np.transpose(rMatrix)
#Loop over all combos that do not include the same image
for i in range(0,500):
    for j in range(0,500):
        if(i==j):
            continue
        else:
            # Resize the thing to make it acceptable by np.dot
            firstIMG = np.array(X[i],None)
            secondIMG = np.array(X[j],None)

            fpX1 = np.dot(rTransp, firstIMG)
            fpX2 = np.dot(rTransp, secondIMG)

            # part1 = abs(np.subtract(fpX1,fpX2))
            # part2 = abs(np.subtract(firstIMG,secondIMG))

            part1 = np.linalg.norm(fpX1 - fpX2)
            part2 = np.linalg.norm(firstIMG - secondIMG)

            part3 = np.divide(part1, part2)
            # print(part3)

            storeEDistances.append(part3)
            storepart1.append(part1)
            storepart2.append(part2)

print("Average for the upper part is")
print(np.mean(storepart1))
# plt.hist(storepart1)
# plt.show()

print("Average for computed values is")
print(np.mean(storeEDistances))

print("Upper values divided by average computed value is")
print(np.mean(storepart1)/np.mean(storeEDistances))

plt.hist(storeEDistances)
plt.show()
# #P is originele set en Q de nieuwe als het goed is?
# p = np.amax(X)
# q = np.amax(y)
#
# randProjXOutputstuff = abs(p*np.array(fP) - q*np.array(fP))/ abs(p - q)




# #
# plt.hist(X)
# plt.show()

print(np.amax(randProjXOutputstuff))

###Part 2
#Dit doen we lekker lelijk gewoon meerdere keren door handmatig de K aan te passen.
fP= np.dot(X,rMatrix)
# train a classifier on the initial values
originalModel = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
originalModel.fit(X_train, y_train)
# evaluate the model and update the list of accuracies
print(metrics.classification_report(originalModel.predict(X_test), y_test))
print(metrics.confusion_matrix(originalModel.predict(X_test),y_test))


# create the random projection
X_trainRand, X_testRand, y_trainRand, y_testRand = train_test_split(fP, y, random_state=30, train_size=0.8)
# train a classifier on the sparse random projection
rpModel = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
rpModel.fit(X_trainRand, y_trainRand)
# evaluate the model and update the list of accuracies
predictLabels = rpModel.predict((X_testRand))
print(metrics.classification_report(predictLabels, y_testRand))
print(metrics.confusion_matrix(predictLabels,y_testRand))
