from scipy.cluster.vq import *
from pylab import *
from numpy import random, array, vstack

class1 = 1.5 * random.randn(100, 2)
# print(class1)
class2 = random.randn(100, 2) + array([5, 5])
# print(class2)

features = vstack((class1, class2))

centroids, variance = kmeans(features, 2)

code, distance = vq(features, centroids)

figure()
ndx = where(code==0)[0]
plot(features[ndx, 0], features[ndx, 1], '*')
ndx = where(code==1)[0]
plot(features[ndx, 0], features[ndx, 1], 'r.')
plot(centroids[:, 0], centroids[:, 1], 'go')
axis('off')
show()

