import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

def price_to_return(X):
    '''return(t) = [price(t)-price(t-1)]/price(t-1)'''
    r_X =  (X[:, 1:] - X[:, :-1])/X[:, :-1]
    return r_X

def dataloader(path = 'snp452-data.mat'):
    '''
    load and preprocess the data
    return:
        X: data matrix, of size (452, 1257)
        X_class: a dict: {class0:data-array0, ..., class9:data-array9}
        C: labels of size (452,)
        cl2co: a dict whose key is the abrr of class name and 
        class2bbr: dict from class name to their abbreviation
        bbr2class: inverse of class2bbr
    '''
    
    class2bbr = {'Consumer Discretionary':'CD', 
                 'Consumer Staples':'CS', 
                 'Energy':'EN', 
                 'Financials':'FI', 
                 'Health Care':'HE', 
                 'Industrials':'IN', 
                 'Information Technology':'IT',
                 'Materials':'MA', 
                 'Telecommunications Services':'TE', 
                 'Utilities':'UT'}
    
    bbr2class = {}
    for k in class2bbr:
        bbr2class[class2bbr[k]] = k
    
    X_index = {'CD':[], 
           'CS':[], 
           'EN':[], 
           'FI':[], 
           'HE':[], 
           'IN':[], 
           'IT':[], 
           'MA':[], 
           'TE':[], 
           'UT':[]}
    
    cl2co = {'CD':[], 
           'CS':[], 
           'EN':[], 
           'FI':[], 
           'HE':[], 
           'IN':[], 
           'IT':[], 
           'MA':[], 
           'TE':[], 
           'UT':[]}

    cl2C = {'CD':0, 
           'CS':1, 
           'EN':2, 
           'FI':3, 
           'HE':4, 
           'IN':5, 
           'IT':6, 
           'MA':7, 
           'TE':8, 
           'UT':9}
    
    mat_data = scio.loadmat(path)
    
    X = mat_data['X']
    X = X.transpose()
    X = price_to_return(X)
    
    C = np.zeros(X.shape[0])
    
    stock = mat_data['stock']
    
    for i, s in enumerate(stock[0]):
        co = s[0][0][0][0].replace('"','')
        na = s[0][0][1][0].replace('"','')
        cl = s[0][0][2][0].replace('"','')
        X_index[class2bbr[cl]].append(i)
        cl2co[class2bbr[cl]].append((co, na))
        C[i] = cl2C[class2bbr[cl]]
    
    X_class = {}
    for k in X_index:
        X_class[k] = X[X_index[k]]
    
    return X, X_class, X_index, C, cl2co, class2bbr, bbr2class


class R_pca:
    '''implementation of RobustPCA
    from https://github.com/dganguli/robust-pca/blob/master/r_pca.py
    '''
    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * self.frobenius_norm(self.D))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.frobenius_norm(self.D)

        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.frobenius_norm(self.D - Lk - Sk)
            iter += 1
            if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
                print('iteration: {0}, error: {1}'.format(iter, err))

        self.L = Lk
        self.S = Sk
        return Lk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True):

        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        print('ymin: {0}, ymax: {1}'.format(ymin, ymax))

        numplots = np.min([n, nrows * ncols])
        plt.figure()

        for n in range(numplots):
            plt.subplot(nrows, ncols, n + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[n, :] + self.S[n, :], 'r')
            plt.plot(self.L[n, :], 'b')
            if not axis_on:
                plt.axis('off')


def accuracy(pred_y, y):
    '''
    calculate accuracy give predicted label pred_y and ground truth y
    return:
        acc: accuracy
        misclass: miclassified ground true (1-10)
        mistarget: to what the ground true is misclassified to (1-10)   
    '''
    mask = np.ones_like(y)*(pred_y==y)
    misclass = (1-mask)*(y+1)
    mistarget = (1-mask)*(pred_y+1)
    acc = mask.mean()
    return acc, misclass, mistarget
    
def confusion_map(misclass, mistarget):
    '''
    print out the pair of (ground true -> misclassified label) pair and
    their frequency
    '''
    rec = {}
    for i in range(len(misclass)):
        c = int(misclass[i])
        t = int(mistarget[i])
        if c!=0 and t!=0:
            if '%d->%d'%(c,t) in rec.keys():
                rec['%d->%d'%(c,t)] += 1
            else:
                rec['%d->%d'%(c,t)] = 1
    print(rec)
    

    