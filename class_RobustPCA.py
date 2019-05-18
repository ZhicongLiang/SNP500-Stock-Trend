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

# use R_pca to estimate the degraded data as L + S, where L is low rank, and S is sparse
Ls = {}
Ss = {}
keys = X_class.keys()
for k in keys:
    print('-----------{}-----------'.format(k))
    D = X_class[k]
    rpca = R_pca(D, mu=1)
    L, S = rpca.fit(max_iter=10000, iter_print=10000)
    Ls[k] = L
    Ss[k] = S
    X[X_index[k]] = L
    
print('Rank of L:', np.linalg.matrix_rank(X))
    
X_train, X_test = X[:300], X[300:]
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
    plt.plot(Ls['IT'][i]+0.3*i)
plt.title('Information Technology (L)')
plt.yticks([])

plt.subplot(313)
for i in range(10):
    plt.plot(Ss['IT'][i]+0.3*i)
plt.title('Information Technology (S)')
plt.yticks([])

# plot the mean of each class
f, ax = plt.subplots()
legend = []

for i, k in enumerate(keys):
    y = np.mean(Ls[k], axis=0)
    ax.plot(y-0.1*i, label=bbr2class[k])
ax.legend(bbox_to_anchor=(1, 1.03))
plt.yticks([])

# find the correlation of each class
#y_mean = np.zeros((10,1257))
#keys = Ls.keys()
#for i,k in enumerate(keys):
#    y_mean[i,:] = np.mean(Ls[k], axis=0)
#corr = np.corrcoef(y_mean)
#print("corr:", corr) 
