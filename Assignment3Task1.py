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


#Standard OML get data stuff
oml.config.apikey = '366585f9ec2435e0f2543d5175e1bac5'
mnist_data = oml.datasets.get_dataset(554) # Download MNIST data
# Get the predictors X and the labels y
X, y = mnist_data.get_data(target=mnist_data.default_target_attribute);
smallerX, _, smallery, _ = train_test_split(X, y, train_size=0.15, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(smallerX, smallery, random_state=30, train_size=0.8)

#Use the following values of $k=45,90,400$ in your experiments.
k=45


###Part 2


#Check without random projection
nn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
nn.fit(X_train,y_train)
print("Training set score: {:.3f}".format(nn.score(X_train, y_train)))
print("Test set score: {:.3f}".format(nn.score(X_test, y_test)))
