import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import time
from sklearn import cluster
from sklearn import metrics
import scipy.cluster.hierarchy as shc


# Récupération des données

def get_data(fichier):
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
    return [datanp, f0, f1]

def check_nb_labels(label):
    compteur = 0;
    list = []
    for i in range(len(label)):
        if label[i] not in list:
            list.append(label[i])
            compteur += 1
            if (compteur == 2):
                return True
    return False

#
#Evaluation du clustering
#
def evaluation_clustering (data, label):
    nb_labels_ok = check_nb_labels(label)
    if (nb_labels_ok):
        silhouette_score = metrics.silhouette_score(data, label, metric='euclidean', sample_size=None, random_state=None,)
        davies_bouldin = metrics.davies_bouldin_score(data, label)
        calinski = metrics.calinski_harabasz_score(data, label)
    else:
        silhouette_score=-1
        davies_bouldin=1000
        calinski=0
    return [silhouette_score, davies_bouldin, calinski]


def print_cluster_optimal_k_means (fichier):
    
    [datanp, f0, f1]=get_data(fichier)
    
    best_k = [2, 2, 2]
    best = [-1, 1000000000, 0]
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

        tps1 = time.time()
        model = cluster.KMeans( n_clusters=i, init='k-means++', n_init='auto')
        model.fit(datanp)
        tps2 = time.time()
        labels = model.labels_
        iteration = model.n_iter_
        plt.scatter(f0, f1, c = labels, s = 8 )
        print ("nb clusters =" ,i , ", nb iter = ", iteration ," , runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms ")
        if (compteur == 0):
            plt.title( "Donnees apres clustering Kmeans pour silhouette")
            print ("Best silhouette score =", best[compteur])
        if (compteur == 1):
            plt.title( "Donnees apres clustering Kmeans pour davies")
            print ("Best davies score =", best[compteur])
        if (compteur == 2):
            plt.title( "Donnees apres clustering Kmeans pour calinski")
            print ("Best calinski score =", best[compteur])
        compteur += 1
        plt.show()
        
def afficher_dendrogramme (fichier):
    
    [datanp, f0, f1]=get_data(fichier)
    
    # Donnees dans datanp
    print("Dendrogramme 'single' donnees initiales")
    linked_mat = shc.linkage(datanp, 'single')
    plt.figure(figsize =( 12 , 12 ))
    shc.dendrogram(linked_mat, orientation = 'top', distance_sort ='descending', show_leaf_counts = False )
    plt.show ()
    
def clustering_hierarchique_distance(fichier, distance):
    
    [datanp, f0, f1]=get_data(fichier)
    
    # set di stance_threshold ( 0 ensures we compute the full tree )
    tps1 = time.time ()
    model = cluster.AgglomerativeClustering( distance_threshold = distance , linkage = 'single' , n_clusters = None )
    model = model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    k = model.n_clusters_
    leaves = model.n_leaves_
    # Affichage clustering
    plt.scatter ( f0 , f1 , c = labels , s = 8 )
    plt.title ( " Resultat du clustering pour une distance donnée" )
    plt.show ()
    print ( "nb clusters = " ,k , " , nb feuilles = " , leaves ," runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
    
    
def clustering_hierarchique_distance2(datanp, distance, method):
    
    # set distance_threshold ( 0 ensures we compute the full tree )
    tps1 = time.time ()
    model = cluster.AgglomerativeClustering( distance_threshold = distance , linkage = method , n_clusters = None )
    model = model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    k = model.n_clusters_
    leaves = model.n_leaves_
    return [labels, tps1, tps2, k, leaves]
    
    
def clustering_hierarchique_cluster (fichier, nb_cluster):
    
    [datanp, f0, f1]=get_data(fichier)
    
    # set the number of clusters
    tps1 = time.time()
    model = cluster.AgglomerativeClustering ( linkage = 'single' , n_clusters = nb_cluster )
    model = model.fit ( datanp )
    tps2 = time.time ()
    labels = model.labels_
    kres = model.n_clusters_
    leaves = model.n_leaves_
    
    # Affichage clustering
    plt.scatter ( f0 , f1 , c = labels , s = 8 )
    plt.title ( " Resultat du clustering pour un nombre de cluster" )
    plt.show ()
    print ( "nb clusters = " ,nb_cluster , " , nb feuilles = " , leaves ," runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )

def clustering_hierarchique_cluster2 (datanp, nb_cluster, method):

    # set the number of clusters
    tps1 = time.time()
    model = cluster.AgglomerativeClustering ( linkage = method, n_clusters = nb_cluster )
    model = model.fit ( datanp )
    tps2 = time.time ()
    labels = model.labels_
    kres = model.n_clusters_
    leaves = model.n_leaves_
    return [labels, tps1, tps2, nb_cluster, leaves]
    
def best_clustering_distance (fichier, method):
    
    [datanp, f0, f1]=get_data(fichier)
    
    best_distance = [0, 0, 0]
    best = [-1, 1000000000, 0]
    for k in np.arange(0.1,5, 0.1):
        
        [labels, tps1, tps2, nb_cluster, leaves] = clustering_hierarchique_distance2(datanp, k, method)
        [silhouette, davies, calinski] = evaluation_clustering(datanp, labels)
        if (silhouette > best[0]):
            best[0] = silhouette
            best_distance[0] = k
        if (davies < best[1]):
            best[1] = davies
            best_distance[1] = k
        if (calinski > best[2]):
            best[2] = calinski
            best_distance[2] = k
    compteur = 0
    for i in best_distance:
        
        [labels, tps1, tps2, nb_cluster, leaves] = clustering_hierarchique_distance2(datanp, i, method)
        plt.scatter(f0, f1, c = labels, s = 8 )
        print ( "best distance = ", i, "nb clusters = " ,nb_cluster , " , nb feuilles = " , leaves ," runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
        if (compteur == 0):
            plt.title( "Donnees apres clustering pour silhouette")
            print ("Best silhouette score =", best[compteur])
        if (compteur == 1):
            plt.title( "Donnees apres clustering pour davies")
            print ("Best davies score =", best[compteur])
        if (compteur == 2):
            plt.title( "Donnees apres clustering pour calinski")
            print ("Best calinski score =", best[compteur])
        compteur += 1
        plt.show()
        
def best_clustering_nb_cluster (fichier, method):
    
    [datanp, f0, f1]=get_data(fichier)
    
    best_nb_cluster = [0, 0, 0]
    best = [-1, 1000000000, 0]
    for k in range(2, 6):
        [labels, tps1, tps2, nb_cluster, leaves] = clustering_hierarchique_cluster2(datanp, k, method)
        [silhouette, davies, calinski] = evaluation_clustering(datanp, labels)
        if (silhouette > best[0]):
            best[0] = silhouette
            best_nb_cluster[0] = k
        if (davies < best[1]):
            best[1] = davies
            best_nb_cluster[1] = k
        if (calinski > best[2]):
            best[2] = calinski
            best_nb_cluster[2] = k
    
    compteur = 0
    for i in best_nb_cluster:
        
        [labels, tps1, tps2, nb_cluster, leaves] = clustering_hierarchique_cluster2(datanp, i, method)
        plt.scatter(f0, f1, c = labels, s = 8 )
        print ("nb clusters = " ,i , " , nb feuilles = " , leaves ," runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
        if (compteur == 0):
            plt.title( "Donnees apres clustering pour silhouette")
            print ("Best silhouette score =", best[compteur])
        if (compteur == 1):
            plt.title( "Donnees apres clustering pour davies")
            print ("Best davies score =", best[compteur])
        if (compteur == 2):
            plt.title( "Donnees apres clustering pour calinski")
            print ("Best calinski score =", best[compteur])
        compteur += 1
        plt.show()
    

#print_cluster_optimal_k_means("xclara")
#afficher_dendrogramme("xclara")
#clustering_hierarchique_distance("cassini", 0.2)
#clustering_hierarchique_cluster("cassini", 3)
#best_clustering_distance("cassini", 'single')
best_clustering_nb_cluster("banana", 'complete')


#0.3583341880308135
    

