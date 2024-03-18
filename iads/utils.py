# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2023-2024, Sorbonne Université
"""


# Fonctions utiles
# Version de départ : Février 2024

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ 

def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    
    desc = np.random.uniform(binf,bsup,(n*2,p))
    label = np.asarray([-1 for i in range(n)] + [+1 for i in range(n)])
    
    return desc, label

# genere_dataset_gaussian:
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    neg = np.random.multivariate_normal(negative_center, negative_sigma,nb_points)
    pos = np.random.multivariate_normal(positive_center, positive_sigma,nb_points)
    desc = np.concatenate( (neg, pos) )
    label = np.asarray([-1 for i in range(nb_points)] + [+1 for i in range(nb_points)])
    
    return desc, label

# plot2DSet:
def plot2DSet(desc,labels):    
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
    # Extraction des exemples de classe -1:
    data_negatifs = desc[labels == -1]
    # Extraction des exemples de classe +1:
    data_positifs = desc[labels == +1]
    
    print("label -1: o_rouge; label +1: x_bleu")
    # Affichage de l'ensemble des exemples :
    plt.scatter(data_negatifs[:,0],data_negatifs[:,1],marker='o', color="red") # 'o' rouge pour la classe -1
    plt.scatter(data_positifs[:,0],data_positifs[:,1],marker='x', color="blue") # 'x' bleu pour la classe +1


# plot_frontiere:
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])


# create_XOR:  
def create_XOR(n, var):
    """ int * float -> tuple[ndarray, ndarray]
        Hyp: n et var sont positifs
        n: nombre de points voulus
        var: variance sur chaque dimension
    """
    neg1 = np.random.multivariate_normal(np.array([0,0]), np.array([[var,0],[0,var]]), n)
    neg2 = np.random.multivariate_normal(np.array([1,1]), np.array([[var,0],[0,var]]), n)
    pos1 = np.random.multivariate_normal(np.array([1,0]), np.array([[var,0],[0,var]]), n)
    pos2 = np.random.multivariate_normal(np.array([0,1]), np.array([[var,0],[0,var]]), n)
    
    descriptions = np.vstack((neg1, neg2, pos1, pos2))
    labels = np.asarray([-1 for i in range(2*n)] + [+1 for i in range(2*n)])
    
    return descriptions, labels
