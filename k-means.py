import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import cluster
#
# Les donnees sont dans datanp ( 2 dimensions )
# f0 : valeurs sur la premiere dimension
# f1 : valeur sur la deuxieme dimension
#
print ("Appel KMeans pour une valeur fixee de k")
tps1 = time.time()
k=3
model = cluster.KMeans( n_clusters=k, init='k-means++')
model.fit(datanp)
tps2 = time.time()
labels = model.labels_
iteration = model.n_iter_
plt.scatter(f0, f1, c = labels, s = 8 )
plt.title( "Donnees apres clustering Kmeans" )
plt.show()
print ("nb clusters =" ,k , ", nb iter = ", iteration ," , runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
