'''Here, we will see whether we could use PCA
as our regression feature to classifier a stock's class'''

from utils import dataloader, accuracy, confusion_map
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

np.random.seed(5)


# prepare data
X, X_class, X_index, Y, cl2co, class2bbr, bbr2class = dataloader()
X_train, X_test = X[:300], X[300:]
Y_train, Y_test = Y[:300], Y[300:]


# PCA
n = 150
pca = PCA(n_components=n)
pca.fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

print('Number of Component:', n)
print('Explain Variance: {:.3f}'.format(sum(pca.explained_variance_ratio_)))


# logistic regression
clf = LogisticRegression(C=60, solver='newton-cg')
clf.fit(X_train, Y_train)
acc_train, *_ = accuracy(clf.predict(X_train), Y_train)
acc_test, misclass, mistarget = accuracy(clf.predict(X_test), Y_test)

print('Train acc:{:.3f}'.format(acc_train))
print('Test acc:{:.3f}'.format(acc_test))


# embedding
f = plt.figure()
plt.scatter(X_test[:,0], X_test[:,1], c=Y_test)
plt.colorbar()
plt.xticks([])
plt.yticks([])

confusion_map(misclass, mistarget)

