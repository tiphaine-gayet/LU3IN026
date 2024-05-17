# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2023-2024, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# ------------------------ 
def dist_euclidienne(X , Y):
    return np.linalg.norm(X - Y )

# ------------------------ 

def normalisation(df):
    df_normalized = df.copy()
    
    # Parcourir chaque colonne du DataFrame
    for column in df_normalized.columns:
        # Trouver la valeur minimale et maximale de la colonne
        min_value = df_normalized[column].min()
        max_value = df_normalized[column].max()
        
        # Normaliser les valeurs de la colonne
        df_normalized[column] = (df_normalized[column] - min_value) / (max_value - min_value)
    
    return df_normalized

# ------------------------ 
    
def centroide(df):
    return np.mean(df, axis=0)


def nouveaux_centroides(Base,U):
    """ Array * dict[int,list[int]] -> DataFrame
        Base : Array contenant la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    d = []
    for i in U.keys():
        d.append(centroide(Base.iloc[U[i]]))
    return np.array(d) 
    
# ------------------------ 
    
def dist_centroides(df_v1, df_v2):
    return dist_euclidienne(centroide(df_v1), centroide(df_v2))
    
# ------------------------ 
    
def initialise_CHA(df) :
    partition = {}  # Initialiser la partition
    
    # Parcourir les index de chaque exemple dans le DataFrame
    for idx in df.index:
        partition[idx] = [idx]  # Affecter chaque exemple à son propre cluster initial
    
    return partition
    
# ------------------------ 
    
def fusionne(dataframe, P0, verbose = False):
    distances = {}
    for i in P0:
        j = i+1
        for j in P0:
            if i < j:
                distances[(i,j)] = dist_centroides(dataframe.iloc[P0[i]],dataframe.iloc[P0[j]])
    # on cherche le couple (i,j) qui correspond à la plus petite distance:
    min_dist = min(distances.values())
    for couple in distances:
        if distances[couple] == min_dist:
            cle_1,cle_2 = couple
            break
    # on fusionne les clusters i et j de P0:
    P1 = P0.copy()
    P1[i+1] = P0[cle_1] + P0[cle_2]
    del P1[cle_1]
    del P1[cle_2]
    # on affiche le résultat si verbose = True
    if verbose:
        print(f"Distance mininimale trouvée entre  [{cle_1}, {cle_2}] : {min_dist}")
    return P1, cle_1, cle_2, min_dist
    
# ------------------------ 
    
import scipy.cluster.hierarchy

def CHA_centroid(DF, verbose=False, dendrogramme=False):
    res = []
    partition = initialise_CHA(DF)
    
    if verbose:
        print("CHA_centroid: clustering hiérarchique ascendant, version Centroid Linkage")
    
    while len(partition) > 1:
        next_partition, key1, key2, dist = fusionne(DF, partition, verbose)
        nb_examples = len(partition[key1]) + len(partition[key2])
        
        if verbose:
            print(f"CHA_centroid: une fusion réalisée de {key1} avec {key2} de distance {dist:.4f}")
            print(f"CHA_centroid: le nouveau cluster contient {nb_examples} exemples")
        
        res.append([key1, key2, dist, nb_examples])
        partition = next_partition
    
    if verbose:
        print("CHA_centroid: plus de fusion possible, il ne reste qu'un cluster unique.")
    
    if dendrogramme:
        plt.figure(figsize=(30, 15))  # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        scipy.cluster.hierarchy.dendrogram(
            res,
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )

        plt.show()
    
    return res

#----------------
def plus_proche(Exe,Centres):
    """ Array * Array -> int
        Exe : Array contenant un exemple
        Centres : Array contenant les K centres
    """
    distances = [(dist_euclidienne(Exe, Centres[i]), i) for i in range(len(Centres))]
    _, dist_min = min(distances)
    return dist_min


def affecte_cluster(Base,Centres):
    """ Array * Array -> dict[int,list[int]]
        Base: Array contenant la base d'apprentissage
        Centres : Array contenant des centroides
    """
    d = {i: [] for i in range(len(Centres))}
    for i in range(0,len(Base)):
        pproche = plus_proche(Base.iloc[i],Centres)
        d[pproche].append(i)
    return d   

def inertie_cluster(Ens):
    """ Array -> float
        Ens: array qui représente un cluster
        Hypothèse: len(Ens)> >= 2
        L'inertie est la somme (au carré) des distances des points au centroide.
    """
    return np.sum(dist_euclidienne(Ens,centroide(Ens))**2) 
    
    
def inertie_globale(Base, U):
    """ Array * dict[int,list[int]] -> float
        Base : Array pour la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    clusters = []
    for i in U.keys():
        clusters.append(Base.iloc[U[i]])
    return sum([inertie_cluster(cluster) for cluster in clusters])

def init_kmeans(K,Ens):
    """ int * Array -> Array
        K : entier >1 et <=n (le nombre d'exemples de Ens)
        Ens: Array contenant n exemples
    """
    return np.array(pd.DataFrame(Ens).sample(n=K))
    

def kmoyennes(K, Base, epsilon, iter_max):
    """ int * Array * float * int -> tuple(Array, dict[int,list[int]])
        K : entier > 1 (nombre de clusters)
        Base : Array pour la base d'apprentissage
        epsilon : réel >0
        iter_max : entier >1
    """
    iter = 0
    # choisir aleatoirement K exemples dans Base comme premiers centres de clusters 
    centres = init_kmeans(K,Base)
    #affecter chaque x de Base au cluster dont il est le plus proche
    affectation = affecte_cluster(Base,centres)
    # mettre a jour les centres des clusters
    new_inertie = inertie_globale(Base,affectation)
    diff = epsilon+1
    #repeter jusqu'a ce que l'inertie ne change plus beaucoup
    while diff>epsilon and iter<iter_max :
        iter+= 1
        centres = nouveaux_centroides(Base,affectation)
        affectation = affecte_cluster(Base,centres)
        old_inertie = new_inertie
        new_inertie = inertie_globale(Base,affectation)
        diff = abs(new_inertie-old_inertie)
        print(f"iteration {iter} Inertie : {new_inertie:.4f} Difference: {diff:.4f}")
        
    return centres, affectation, new_inertie


#--------------------------------------------

# Fonction pour calculer la distance maximale entre les points d'un cluster
def max_distance_in_cluster(cluster_points):
    max_dist = 0
    n = len(cluster_points)
    for i in range(n):
        for j in range(i + 1, n):
            dist = dist_euclidienne(cluster_points.iloc[i], cluster_points.iloc[j])
            if dist > max_dist:
                max_dist = dist
    return max_dist

# Fonction pour calculer la codistance d'une partition
def codistance_partition(clusters, data):
    codistance = 0
    for indices in clusters.values():
        cluster_points = data.iloc[indices]
        if len(cluster_points) > 1:
            codistance += max_distance_in_cluster(cluster_points)
    return codistance


# Fonction pour calculer la coinertie d'un cluster
def coinertie_cluster(cluster_points, centroid):
    total_inertia = 0
    for i in range(len(cluster_points)):
        total_inertia += dist_euclidienne(cluster_points.iloc[i], centroid)**2
    return total_inertia

# Fonction pour calculer la coinertie de la partition
def coinertie_partition(clusters,centroids, data):
    coinertie = 0
    for k, indices in clusters.items():
        cluster_points = data.iloc[indices]
        if len(cluster_points) > 1:
            coinertie += coinertie_cluster(cluster_points, centroids[k])
    return coinertie


def calcule_centroides(clusters, data):
    centroides = {}
    for key, indices in clusters.items():
        centroides[key] = centroide(data.iloc[indices])
    return centroides

def semin_partition(centroids):
    min_dist = float('inf')
    K = len(centroids)
    for i in range(K):
        for j in range(i + 1, K):
            dist = dist_euclidienne(centroids[i], centroids[j])
            if dist < min_dist:
                min_dist = dist
    return min_dist

def DUNN(clusters, centroids, data) :
    return codistance_partition(clusters, data)/semin_partition(centroids)

def XIEBENI(clusters,centroids, data) :
    return coinertie_partition(clusters,centroids, data)/semin_partition(centroids)