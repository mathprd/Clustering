import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import time
from sklearn import cluster
from sklearn import metrics


#
#Evaluation du clustering
#
def evaluation_clustering (data, label):
    silhouette_score = metrics.silhouette_score(data, label, metric='euclidean', sample_size=None, random_state=None,)
    davies_bouldin = metrics.davies_bouldin_score(data, label)
    calinski = metrics.calinski_harabasz_score(data, label)
# =============================================================================
#     print ("silhouette score = ", silhouette_score)
#     print ("davies bouldin score = ", davies_bouldin)
#     print ("calinski harbasz score = ", calinski)
# =============================================================================
    return [silhouette_score, davies_bouldin, calinski]


def print_cluster_optimal (fichier):
    #
    # =============================================================================
    # Parser un fichier de donnees au format arff
    # data est un tableau d ’ exemples avec pour chacun
    # la liste des valeurs des features
    # Dans les jeux de donnees consideres :
    # il y a 2 features ( dimension 2 )
    # Ex : [[ - 0 . 499261 , -0 . 0612356 ] ,
    # [ - 1 . 51369 , 0 . 265446 ] ,
    # [ - 1 . 60321 , 0 . 362039 ] , .....
    # ]
    # Note : chaque exemple du jeu de donnees contient aussi un
    # numero de cluster . On retire cette information
    # =============================================================================
    path = './clustering-benchmark-master/src/main/resources/datasets/artificial/'
    databrut = arff.loadarff (open(path + fichier + ".arff", 'r') )
    datanp = [ [ x[ 0 ] ,x[ 1 ] ] for x in databrut[ 0 ] ]

    # Affichage en 2D
    # Extraire chaque valeur de features pour en faire une liste
    # Ex pour f0 = [ - 0 . 499261 , -1 . 51369 , -1 . 60321 , ...]
    # Ex pour f1 = [ - 0 . 0612356 , 0 . 265446 , 0 . 362039 , ...]
    f0 = [x[0] for x in datanp] # tous les elements de la premiere colonne
    f1 = [x[1] for x in datanp] # tous les elements de la deuxieme colonne
# =============================================================================
#     plt.scatter(f0, f1, s = 8)
#     plt.title ("Donnees initiales")
#     plt.show()
# =============================================================================
    best_k = [2, 2, 2]
    best = [0, 1000000000, 0]
    for k in range(2,6):
        #
        # Les donnees sont dans datanp ( 2 dimensions )
        # f0 : valeurs sur la premiere dimension
        # f1 : valeur sur la deuxieme dimension
        #
        model = cluster.KMeans( n_clusters=k, init='k-means++', n_init='auto')
        model.fit(datanp)
        labels = model.labels_
        [silhouette, davies, calinski] = evaluation_clustering(datanp, labels)
        if (silhouette > best[0]):
            best[0] = silhouette
            best_k[0] = k
        if (davies < best[1]):
            best[1] = davies
            best_k[1] = k
        if (calinski > best[2]):
            best[2] = calinski
            best_k[2] = k
    
    compteur = 0
    for i in best_k:
        #
        # Les donnees sont dans datanp ( 2 dimensions )
        # f0 : valeurs sur la premiere dimension
        # f1 : valeur sur la deuxieme dimension
        #
        tps1 = time.time()
        model = cluster.KMeans( n_clusters=i, init='k-means++', n_init='auto')
        model.fit(datanp)
        tps2 = time.time()
        labels = model.labels_
        iteration = model.n_iter_
        plt.scatter(f0, f1, c = labels, s = 8 )
        plt.title( "Donnees apres clustering Kmeans pour le cas")
        plt.show()
        print ("nb clusters =" ,i , ", nb iter = ", iteration ," , runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms ")
        if (compteur == 0):
            print ("Best silhouette score =", best[compteur])
        if (compteur == 1):
            print ("Best davies score =", best[compteur])
        if (compteur == 2):
            print ("Best calinski score =", best[compteur])
        compteur += 1
        

print_cluster_optimal("xclara")
    

