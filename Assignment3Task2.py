from preamble import *
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import scipy
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 100 # This controls the size of your figures
# Comment out and restart notebook if you only want the last output of each cell.
InteractiveShell.ast_node_interactivity = "all"

#Standard OML get data stuff
oml.config.apikey = '366585f9ec2435e0f2543d5175e1bac5'
mnist_data = oml.datasets.get_dataset(554) # Download MNIST data
# Get the predictors X and the labels y
X, y = mnist_data.get_data(target=mnist_data.default_target_attribute);

smallerX, _, smallery, _ = train_test_split(X, y, train_size=0.15, stratify=y)

# Find all the fours in the dataset
fourindexes = np.where(smallery == 4)
# print(yfour)
Xfours = smallerX[fourindexes]
print(Xfours)
yfours = smallery[fourindexes]


X_train, X_test, y_train, y_test = train_test_split(smallerX, smallery, random_state=30, train_size=0.8)

# build a list of figures on a 5x5 grid for plotting
def buildFigure5x5(fig, subfiglist):

    for i in range(0 ,25):
        pixels = np.array(subfiglist[i], dtype='float')
        pixels = pixels.reshape((28, 28))
        a=fig. add_subplot(5,5,i+1)
        imgplot =plt. imshow(pixels, cmap='gray_r')
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
    return

#DEMO STUFF:
# find the first 25 instances with label '4' and plot them
imgs = np.empty([25, 28*28], dtype='float')
j=0
for i in range(0,len(X)):
    if(y[i] == 4) and j < 25:
        imgs[j] = np.array(X[i], dtype='float')
        j += 1
#
# buildFigure5x5(plt.figure(1),imgs)
# plt. show()


#Get a PCA with the first two principal components
pca = PCA(n_components=2,)
pca.fit(Xfours)
Xfours = pca.transform(Xfours)
#Plot the results in a scatterplot
plt.scatter(Xfours[:,0],Xfours[:,1])
# plt.show()

#Reconstructing 25 points on a 5x5 grid.
xRow = [750, 375, 0, -375, -750]
yRow = [750, 375, 0, -375, -750]

for i in xRow:
    for j in yRow:
        plt.scatter(i,j,color='r')
plt.grid(b=True, which='major', color='black', linestyle='--',)

plt.show()

count = 1
#Get dem pixel arts --> this part doesn't seem right.
for i in xRow:
    for j in yRow:
        usedValue = pca.inverse_transform(i,j)
        pixels = np.array(usedValue[0], dtype='float')
        pixels = pixels.reshape((28, 28))
        a = plt.figure(2).add_subplot(5, 5, count)
        imgplot = plt.imshow(pixels, cmap='gray_r')
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
        count += 1
plt.show()


count = 1
#Deel B je weet.
for i in xRow:
    for j in yRow:
        hoi = scipy.spatial.distance.cdist(Xfours, [(i,j)])
        minimalDistance = np.argmin(hoi)
        pixels = np.array(smallerX[fourindexes][minimalDistance], dtype='float')
        pixels = pixels.reshape((28, 28))
        a = plt.figure(3).add_subplot(5, 5, count)
        imgplot = plt.imshow(pixels, cmap='gray_r')
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
        count += 1

plt.show()

#Deel C tijd
#Get the mean Four image
meanFour = np.mean(smallerX[fourindexes], axis=0)
pixels = np.array(meanFour, dtype='float')
pixels = pixels.reshape((28, 28))
imgplot = plt.imshow(pixels, cmap='gray_r')
plt.show()

#Obtain first principal component image
meanFirstPrincip = np.mean(Xfours[:,0], axis=0)
usedValue = pca.inverse_transform(meanFirstPrincip)
pixels = np.array(usedValue[0], dtype='float')
pixels = pixels.reshape((28, 28))
imgplot = plt.imshow(pixels, cmap='gray_r')
plt.show()

#Obtain second principal component image
meanSecondPrincip = np.mean(Xfours[:,1], axis=0)
usedValue = pca.inverse_transform(meanSecondPrincip)
pixels = np.array(usedValue[1], dtype='float')
pixels = pixels.reshape((28, 28))
imgplot = plt.imshow(pixels, cmap='gray_r')
plt.show()
