# -*- coding: utf-8 -*-
"""
Manual PCA implementation (-> sklearn.decomposition.PCA)
"""

import copy
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA

# dataset to perform PCA on
# input format: array with parameters per column, data per row
# same amount of data for each parameter
init_data = datasets.load_iris().data

# standardise data?
# standardising without scaling will just center the data
# note: the sklearn algorithm centers the data but does not scale it
standardise_data = True
scale_data       = False

# deep copy of the initial data to work on
data = copy.deepcopy(init_data)

# 1. statistical analysis
means     = [0 for k in range(data.shape[1])]
variances = [0 for k in range(data.shape[1])]
stdevs    = [0 for k in range(data.shape[1])]
for i in range(data.shape[1]):
    # step 1 - mean
    means[i] = sum(data[:,i])/float(len(data[:,i]))
    # step 2 - variance
    deviation = 0
    for val in data[:,i]:
        deviation += (val-means[i])**2
    variances[i] = deviation/float(len(data[:,i])-1)
    # step 3 - standard deviation
    stdevs[i] = np.sqrt(variances[i])

# 2. data standardisation
if standardise_data:
    # z = x-mean || z = (x-mean)/sd
    for i in range(data.shape[1]):
        for j in range(len(data[:,i])):
            if scale_data:
                data[j,i] = (data[j,i]-means[i])/stdevs[i]
            else:
                data[j,i] = data[j,i]-means[i]

# 3. covariance matrix
covar = np.zeros((data.shape[1],data.shape[1]))
for i in range(data.shape[1]):
    for j in range(i+1):
        dev = 0
        for k in range(len(data[:,i])):
            if standardise_data:
                # standardised data - mean=0
                dev += data[k,i]*data[k,j]
            else:
                dev += (data[k,i]-means[i])*(data[k,j]-means[j])
        covar[i,j] = dev/float(len(data[:,i])-1)
        covar[j,i] = covar[i,j]

# 4. eigen decomposition
# characteristic equation of the covar matrix => eigenvalues/eigenvector
# eigenvectors = principal components
# eigenvalues = component variances
# not doing this part manually
eigenvalues,eigenvectors = np.linalg.eig(covar)

# 5. PC prints
pc1     = eigenvectors[:,0]
pc1_var = eigenvalues [0]
print(f"The first  principal component  carries {round(100*pc1_var/sum(eigenvalues),1)}% of the variance")
pc2     = eigenvectors[:,1]
pc2_var = eigenvalues [1]
print(f"The second principal component  carries {round(100*pc2_var/sum(eigenvalues),1)}% of the variance")
pc3     = eigenvectors[:,2]
pc3_var = eigenvalues [2]
print(f"The third  principal component  carries {round(100*pc3_var/sum(eigenvalues),1)}% of the variance")

print(f"The remaining        components carry   {round(100*(sum(eigenvalues)-pc1_var-pc2_var-pc3_var)/sum(eigenvalues),1)}% of the variance")
print()

# 6. data reorganisation along the first three components
feat_vect = np.vstack((pc1,pc2,pc3))
pca_data  = np.transpose(np.matmul(feat_vect, np.transpose(data)))

# 7. check against sklearn PCA
X = PCA(n_components=3).fit_transform(init_data)
if X.all() == pca_data.all():
    print('PCA fits the sklearn function :)')
else:
    print('PCA does not fit the sklearn function :(')