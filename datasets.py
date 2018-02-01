import numpy as np
from sklearn.model_selection import train_test_split


class IHDP(object):
    def __init__(self, path_data="datasets/IHDP/csv", replications=10):
        self.path_data = path_data
        self.replications = replications
        # which features are binary
        self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        # which features are continuous
        self.contfeats = [i for i in xrange(25) if i not in self.binfeats]

    def __iter__(self):
        for i in xrange(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            t, y, y_cf = data[:, 0], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            yield (x, t, y), (y_cf, mu_0, mu_1)

    def get_train_valid_test(self):
        for i in xrange(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            # this binary feature is in {1, 2}
            x[:, 13] -= 1
            idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)
            itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)
            train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
            valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
            yield train, valid, test, self.contfeats, self.binfeats


class SYNData(object):
    def __init__(self, path_data="datasets/IHDP/csv", replications=10):
        self.path_data = path_data
        self.replications = replications
        # which features are binary
        # self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        self.binfeats = [0]
        # which features are continuous
        # self.contfeats = [i for i in xrange(25) if i not in self.binfeats]
        self.contfeats = []

        self.generate_data()

    def generate_data(self):

        data_size = 10000
        self.data_size = data_size

        self.H = np.random.binomial(1, 0.5, data_size)
        self.T = np.random.binomial(1, 0.5*self.H+0.25)
        self.Z = np.random.binomial(1, 0.6*self.T+0.2)
        self.Z_cf = np.random.binomial(1, 0.6 * (1-self.T) + 0.2)

        self.X = np.random.normal(loc=self.Z, scale=np.square(25*self.Z+9*(1-self.Z)))
        self.Y = np.random.binomial(1, 1/(1+np.exp(-3*(self.H+2*(2*self.Z-1)))))
        self.Y_cf = np.random.binomial(1, 1 / (1 + np.exp(-3 * (self.H + 2 * (2 * self.Z_cf - 1)))))
        #E[Y0|Input] = #E[Y0|H=0]
        mu0_0 = (self.Y[np.where(np.logical_and(self.T == 0, self.H == 0))].sum()+self.Y_cf[np.where(np.logical_and(self.T == 1, self.H == 0))].sum())\
                     /np.float32(self.H == 0).sum()
        mu1_0 = (self.Y_cf[np.where(np.logical_and(self.T == 0, self.H == 0))].sum() + self.Y[np.where(np.logical_and(self.T == 1, self.H == 0))].sum()) \
                     /np.float32(self.H == 0).sum()
        mu0_1 = (self.Y[np.where(np.logical_and(self.T == 0, self.H == 1))].sum()+self.Y_cf[np.where(np.logical_and(self.T==1,self.H==1))].sum())\
                     /np.float32(self.H==1).sum()
        mu1_1 = (self.Y_cf[np.where(np.logical_and(self.T==0,self.H==1))].sum() + self.Y[
            np.where(np.logical_and(self.T==1,self.H==1))].sum()) \
                     /np.float32(self.H == 1).sum()
        self.mu0 = (1-self.H)*mu0_0 + self.H*mu0_1
        self.mu1 = (1-self.H) * mu1_0 + self.H * mu1_1
        self.ITE = self.mu1-self.mu0
        self.ATE = self.ITE.mean()
        xm, xs = np.mean(self.X), np.std(self.X)
        self.X = (self.X - xm) / xs
        return

    def __iter__(self):
        assert False
        for i in xrange(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            t, y, y_cf = data[:, 0], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            yield (x, t, y), (y_cf, mu_0, mu_1)

    def get_train_valid_test(self):
        for i in xrange(self.replications):
            rep_size = self.data_size/self.replications
            startidx = i * rep_size
            stopidx = (i+1) * rep_size
            t = self.T[startidx:stopidx][:, np.newaxis]
            y = self.Y[startidx:stopidx][:, np.newaxis]
            y_cf = self.Y_cf[startidx:stopidx][:, np.newaxis]
            mu_0 = self.mu0[startidx:stopidx][:, np.newaxis]
            mu_1 = self.mu1[startidx:stopidx][:, np.newaxis]
            x = self.X[startidx:stopidx][:, np.newaxis]

            idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)
            itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)
            train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
            valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
            yield train, valid, test, self.contfeats, self.binfeats

