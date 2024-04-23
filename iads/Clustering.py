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
    
def fusionne(DF, P0, verbose = False):
    dist_min = math.inf
    key1, key2 = 0, 0

    for C1 in P0 :
        for C2 in P0 :
            dist = dist_centroides(DF.values[P0[C1]], DF.values[P0[C2]])
            if C1!=C2 and dist < dist_min :
                dist_min = dist
                key1, key2 = C1, C2
    if verbose : print(f"fusionne: distance mininimale trouvée entre  [{key1}, {key2}]  =  {dist_min}") 
    if verbose : print(f"fusionne: les 2 clusters dont les clés sont  [{key1}, {key2}]  sont fusionnés") 
    
    P1 = P0.copy()
    new_key = max(P1)+1
    P1[new_key] = P1[key1] + P1[key2]
    if verbose : print(f"fusionne: on crée la  nouvelle clé {new_key}  dans le dictionnaire.") 
    P1.pop(key1)
    P1.pop(key2)
    if verbose : print(f"fusionne: les clés de  [{key1}, {key2}]  sont supprimées car leurs clusters ont été fusionnés.") 

    return (P1, key1, key2, dist_min)
    
# ------------------------ 
    
import scipy.cluster.hierarchy

def CHA_centroid(DF, verbose=False, dendrogramme=False):
    res = []
    partition = initialise_CHA(DF)
    
    if verbose: print("CHA_centroid: clustering hiérarchique ascendant, version Centroid Linkage")
    while len(partition)!=1 :
        partition, key1, key2, dist = fusionne(DF, partition, verbose)
        nb_examples = len(partition[max(partition)])
        if verbose :
            print(f"CHA_centroid: une fusion réalisée de  {key1}  avec  {key2} de distance  {dist:.4f}")
            print(f"CHA_centroid: le nouveau cluster contient  {nb_examples}  exemples")
        res.append([key1, key2, dist, nb_examples])
    if verbose: print("CHA_centroid: plus de fusion possible, il ne reste qu'un cluster unique.")

    if dendrogramme :
        # Paramètre de la fenêtre d'affichage: 
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            res, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )

        # Affichage du résultat obtenu:
        plt.show()
    
    return res
    
