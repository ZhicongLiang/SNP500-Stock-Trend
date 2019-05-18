'''Here, we will see whether we could use RobustPCA
as our regression feature to classifier a stock's class'''

from utils import dataloader, accuracy, R_pca, confusion_map
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

np.random.seed(5)

# prepare data
X, X_class, X_index, Y, cl2co, class2bbr, bbr2class = dataloader()

print('Rank of X:', np.linalg.matrix_rank(X))


# use R_pca to estimate the degraded data as L + S, where L is low rank, and S is sparse
rpca = R_pca(X, mu=1, lmbda=(1257)**-0.5)
L, S = rpca.fit(max_iter=10000, iter_print=100)
print('Rank of L:', np.linalg.matrix_rank(L))
print('Rank of S:', np.linalg.matrix_rank(S))
print('% sparsity of L:', len(L[abs(L)<1e-7*np.linalg.norm(X, ord='fro')])/np.multiply(*L.shape))
print('% sparsity of S:', len(S[abs(S)<1e-7*np.linalg.norm(X, ord='fro')])/np.multiply(*S.shape))

X_train, X_test = L[:300], L[300:]
Y_train, Y_test = Y[:300], Y[300:]


# logistic regression
clf = LogisticRegression(C=30, solver='newton-cg')
clf.fit(X_train, Y_train)
acc_train, *_ = accuracy(clf.predict(X_train), Y_train)
acc_test, misclass, mistarget = accuracy(clf.predict(X_test), Y_test)

print('Train acc:{:.3f}'.format(acc_train))
print('Test acc:{:.3f}'.format(acc_test))

confusion_map(misclass, mistarget)


# plot L and S of 10 Information Technology stocks
f1 = plt.figure(figsize=(9,12))
plt.subplot(311)
for i in range(10):
    plt.plot(X_class['IT'][i]+0.3*i)
plt.title('Information Technology')
plt.yticks([])

plt.subplot(312)
for i in range(10):
    plt.plot(L[X_index['IT'][i]]+0.3*i)
plt.title('Low-rank Component (L)')
plt.yticks([])

plt.subplot(313)
for i in range(10):
    plt.plot(S[X_index['IT'][i]]+0.3*i)
plt.title('Sparse Component (S)')
plt.yticks([])
#f1.savefig('figs/IT-decompose.jpg', dpi=300)


# plot the mean of each class
keys = X_index.keys()
f, ax = plt.subplots()
legend = []
for i, k in enumerate(keys):
    y = L[X_index[k]]
    y = y.mean(axis=0)
    ax.plot(y-0.1*i, label=bbr2class[k])
ax.legend(bbox_to_anchor=(1, 1.03))
plt.yticks([])

# find the correlation of each class
#y_mean = np.zeros((10,1257))
#keys = X_index.keys()
#for i,k in enumerate(keys):
#    y_mean[i,:] = np.mean(L[X_index[k]], axis=0)
#corr = np.corrcoef(y_mean)
#print("corr:", corr)  
