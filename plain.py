'''Here, we will see whether we could use original
as our regression feature to classifier a stock's class'''

from utils import dataloader, accuracy, confusion_map
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)



# prepare data
X, X_class, X_index, Y, cl2co, class2bbr, bbr2class = dataloader()

X_train, X_test = X[:300], X[300:]
Y_train, Y_test = Y[:300], Y[300:]

clf = LogisticRegression(C=30, solver='newton-cg')
clf.fit(X_train, Y_train)
acc_train,*_ = accuracy(clf.predict(X_train), Y_train)
acc_test, misclass, mistarget = accuracy(clf.predict(X_test), Y_test)

print('Train acc:{:.3f}'.format(acc_train))
print('Test acc:{:.3f}'.format(acc_test))

confusion_map(misclass, mistarget)


# visual some stocks
f1 = plt.figure(figsize=(9,7))
plt.subplot(211)
for i in range(10):
    plt.plot(X_class['IT'][i]+0.3*i)
plt.title('Information Technology')
plt.yticks([])

plt.subplot(212)
for i in range(10):
    plt.plot(X_class['HE'][i]+0.3*i)
plt.title('Health Care')
plt.yticks([])
#f1.savefig('figs/IT-v.s.-HE.jpg', dpi=200)


# calculate the frequency of each classes
f, ax = plt.subplots(figsize=(6,3.5))
num = []
keys = X_class.keys()
for k in keys:
    num.append(X_class[k].shape[0])
ax.bar(keys, num)
plt.ylabel('frequency')
for i, v in enumerate(num):
    ax.text(i-0.2, v + 0.5, str(v), color='blue', fontweight='bold')
#f.savefig('figs/frequency.jpg', dpi=200)


# plot the mean of each class
f, ax = plt.subplots()
legend = []
keys = X_class.keys()
for i, k in enumerate(keys):
    y = np.mean(X_class[k], axis=0)
    ax.plot(y-0.1*i, label=bbr2class[k])
ax.legend(bbox_to_anchor=(1, 1.03))
plt.yticks([])


# find the correlation of each class
#y_mean = np.zeros((10,1257))
#keys = X_class.keys()
#for i,k in enumerate(keys):
#    y_mean[i,:] = np.mean(X_class[k], axis=0)
#corr = np.corrcoef(y_mean)
#print("corr:", corr)

