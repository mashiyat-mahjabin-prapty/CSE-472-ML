import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
import sys

np.random.seed(1)

# https://stats.stackexchange.com/a/240492
# https://davidvandebunte.gitlab.io/executable-notes/notes/se/relationship-between-svd-and-pca.html

def low_rank_approximation(mat, k):
    u, s, v = np.linalg.svd(mat)
    S = np.diag(s)
    mat_approx = np.dot(u[:,:k], S[:k,:k])
    return mat_approx

# define gmm class
class GMM:
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = int(max_iter)
        self.mean = None
        self.cov = None
        self.pi = None
        
    def initialize(self):
        # initialize the mean, covariance and pi
        self.mean = np.random.rand(self.k, self.dim)
        self.cov = np.array([np.eye(self.dim)] * self.k)
        self.pi = np.array([1.0 / self.k] * self.k)
    
    def e_step(self, x):
        self.resp = np.zeros((self.k, len(x)))
        for i in range(self.k):
            self.resp[i] = self.pi[i] * multivariate_normal.pdf(x, self.mean[i], self.cov[i], allow_singular=True)
        self.resp /= self.resp.sum(0)
    
    def m_step(self, x):
        Nk = self.resp.sum(1)
        epsilon = 1e-10
        self.mean = np.dot(self.resp, x) / (Nk[:, None] + epsilon)
        for i in range(self.k):
            x_mean = x - self.mean[i]
            self.cov[i] = np.dot(self.resp[i] * x_mean.T, x_mean) / (Nk[i] + epsilon)
        self.pi = Nk / len(x)
    
    def log_likelihood(self, x):
        # calculate the log likelihood
        ll = np.zeros(len(x))
        for j in range(self.k):
            ll += self.pi[j] * multivariate_normal.pdf(x, self.mean[j], self.cov[j], allow_singular=True)
        return np.log(ll).sum()
    
    def fit(self, x):
        self.dim = x.shape[1]
        prev_ll = 0
        self.initialize()
        for i in range(self.max_iter):
            self.e_step(x)
            self.m_step(x)
            # print(self.mean)
            # print(self.cov)
            # print(self.pi)
            # print(self.log_likelihood(x))
            # print('-----------------------')
            if abs(self.log_likelihood(x) - prev_ll) <= 1e-2:
                break
            prev_ll = self.log_likelihood(x)
            if i % 10 == 0:
                print('Iteration: ', i)
                print('The log likelihood is: ', self.log_likelihood(x))
        print('The log likelihood after convergence is: ', self.log_likelihood(x))
    
    def plot(self, x, p):
        # plot the data points and the clusters
        self.e_step(x)
        cluster = self.resp.argmax(axis=0)
        # print(cluster.shape)
        # print(cluster[:5])
        plt.figure(figsize=(10, 10))
        plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=cluster)
        plt.xlabel('Dim1')
        plt.ylabel('Dim2')
        plt.savefig('1805117_'+str(p)+'_clusters'+infile+'.png')
        plt.show()
    
# take the file name from command line argument
n = len(sys.argv)
if n != 2:
    print('Please provide the file name as command line argument')
    exit()
infile = sys.argv[1]
# infile = "6D_data_points"
# Read the data from text file and store in a datafram
df = pd.read_csv(infile + '.txt', sep=',', header=None)

# standardize the data
df = (df - df.mean()) / df.std()

if df.shape[1] <= 2:
    print('The data points are not enough to perform dimensionality reduction')
else:
    i = df.shape[1] - 1
    # perform dim red one dim by one dim
    while i >= 2:
        mat_approx = low_rank_approximation(df, i)
        # print(mat_approx)
        i -= 1
        df = pd.DataFrame(mat_approx)

print(df.head())
# name the columns
for i in range(df.shape[1]):
    df.rename(columns={i: 'Dim'+str(i+1)}, inplace=True)
# Plot the data points
plt.figure(figsize=(10, 10))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
plt.xlabel('Dim1')
plt.ylabel('Dim2')
plt.savefig('1805117_scatter_plot_'+infile+'.png')
plt.show()

# call the gmm class and fit the data
ll_vs_K = []
for p in range(3, 9):
    max_ll = -np.inf
    for q in range(5):
        print('K: ', p, ' Iteration: ', q)
        gmm = GMM(p, 500)
        gmm.fit(df)
        if gmm.log_likelihood(df) > max_ll:
            max_ll = gmm.log_likelihood(df)
            best_gmm = gmm
    best_gmm.plot(df, p)
    print('The best log likelihood is: ', max_ll)
    ll_vs_K.append(max_ll)

# plot the log likelihood vs K
plt.figure(figsize=(10, 10))
plt.plot(range(3, 9), ll_vs_K, marker='o')
plt.xlabel('K')
plt.ylabel('Log likelihood')
plt.savefig('1805117_log_likelihood'+infile+'.png')
plt.show()


# best K = 3, 4, 4, 5
# plot the data points and the clusters
# gmm = GMM(4, 500)
# gmm.fit(df)
# gmm.plot(df)