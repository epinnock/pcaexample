__author__ = 'EJ'

'''
Principal Component Analysis tutorial taken from http://sebastianraschka.com/Articles/2014_pca_step_by_step.html#introduction
hint: CTRL+Q will bring up documentation on highlighted text
'''

import numpy as np
import pylab
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

np.random.seed(1)
#create 2 class of data points centered around different means for the dimensions x,y,z

#class1_sample is centered around [0,0,0]
mu_vec1= np.array([0,0,0])
 #covariance matrix see: http://stattrek.com/matrix-algebra/covariance-matrix.aspx
cov_mat1=np.array([[1,0,0],[0,1,0],[0,0,1]])
class1_sample=np.random.multivariate_normal(mu_vec1,cov_mat1,20).T
assert class1_sample.shape==(3,20),"The shape is wrong should be 3x20"
print 'class 1'
print class1_sample

#class2_sample is centered around[1,1,1]
mu_vec2=np.array([1,1,1])
cov_mat2=np.array([[1,0,0],[0,1,0],[0,0,1]])
class2_sample=np.random.multivariate_normal(mu_vec2,cov_mat2,20).T
assert class2_sample.shape==(3,20),"The shape is wrong should be 3x20"
print 'class 2'
print class2_sample

def showgraph():
    #Create a 3d graph of the two classes
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams['legend.fontsize'] = 10
    ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:],
        'o', markersize=8, color='blue', alpha=0.5, label='class1')
    ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:],
        '^', markersize=8, alpha=0.5, color='red', label='class2')

    plt.title('Samples for class 1 and class 2')
    ax.legend(loc='upper right')

    plt.show()

#showgraph()

#compile data set into 1 large batch
print 'class 1'
print class1_sample
print 'class 2'
print class2_sample
class_total_set=np.concatenate((class1_sample,class2_sample),axis=1)
assert class_total_set.shape==(3,40),"The shape is wrong should be 3 x 40"
print 'class_total_set'
print class_total_set
#2. Computing the d-dimensional mean vector
#compute the mean vector for each of the dimensions(x,y,z)
mean_x=np.mean(class_total_set[0,:])
print 'mean x : %f'%mean_x
mean_y=np.mean(class_total_set[1,:])
print ' mean y : %f'%mean_y
mean_z=np.mean(class_total_set[2,:])
print 'mean z :%f'%mean_z

mean_vector=np.array([[mean_x],[mean_y],[mean_z]])
print ('mean vector \n',mean_vector)



'''
3. a) Computing the Scatter Matrix
'''
scatter_matrix = np.zeros((3,3))
for i in range(class_total_set.shape[1]):
    scatter_matrix += (class_total_set[:,i].reshape(3,1) - mean_vector).dot(
        (class_total_set[:,i].reshape(3,1) - mean_vector).T)
print('Scatter Matrix:\n', scatter_matrix)

'''
3. b) Computing the Covariance Matrix (alternatively to the scatter matrix)
again see: http://stattrek.com/matrix-algebra/covariance-matrix.aspx
'''
cov_mat_all=np.cov(class_total_set)
print ('Covariance Matrix \n',cov_mat_all)

'''
4. Computing eigenvectors and corresponding eigenvalues
An EIGENVECTOR is a vector that does not change direction under a linear transformation
and the EIGENVALUE is the non zero amount by which the EIGENVECTOR is scaled
For mor on eigenvectors see: https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors
'''
eig_val_sc, eig_vec_sc=np.linalg.eig(scatter_matrix)
eig_val_cov, eig_vec_cov=np.linalg.eig(cov_mat_all)

print ("Eigenvector-Covariance unshaped:\n", eig_vec_cov)
print ("Eigenvalue-Covariance :\n", eig_val_cov)


for i in range(len(eig_val_sc)):
    eigvec_sc = eig_vec_sc[:,i].reshape(1,3).T
    eigvec_cov=eig_vec_cov[:,i].reshape(1,3).T
    assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvectors are not identical'

print ("Eigenvector-Covariance shaped: \n",eigvec_cov)

'''
5.1. Sorting the eigenvectors by decreasing eigenvalues
'''

'''
check to see that eigenvectors have a unit lenght of 1
'''
for vec in eig_vec_sc:
    np.testing.assert_array_almost_equal(1.0,np.linalg.norm(vec))


''' group eigenvalues and eigenvectors into tuple list'''
eig_pair=[]
for i in range(len(eig_val_sc)):
    eig_pair.append((np.abs(eig_val_sc[i]),eig_vec_sc[:,i]))

#sort by decreasing value
eig_pair.sort()
eig_pair.reverse()
for i in eig_pair:
    print i

'''
Combine the two eigenvectors with the higest egienvalues to construct an eigenvector matrix
'''
matrix_w= np.hstack((eig_pair[0][1].reshape(3,1),eig_pair[1][1].reshape(3,1)))
print ('Matrix W:\n',matrix_w)

'''
Transforming the samples onto the new matrix
'''
class_total_set_transformed=matrix_w.T.dot(class_total_set)
assert class_total_set_transformed.shape==(2,40),"The shape is wrong should be 2x40"

def showTransformed():
    #plot class1
    plt.plot(class_total_set_transformed[0,0:20],class_total_set_transformed[1,0:20],'o',markersize=7,color='blue',alpha=0.5,label='class1')
    #plot class2
    plt.plot(class_total_set_transformed[0,20:40],class_total_set_transformed[1,20:40],'^',markersize=7,color='red',alpha=0.5,label='class2')
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('Transformed Total Dataset')
    plt.show()
showTransformed()