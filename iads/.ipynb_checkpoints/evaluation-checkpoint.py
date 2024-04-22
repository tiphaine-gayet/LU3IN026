# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2023-2024, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd
import copy


def crossval(X, Y, n_iterations, iteration):
    #############
    # A COMPLETER
    #############  
    start_idx = iteration*len(X)//n_iterations
    end_idx =(iteration+1)*len(X)//n_iterations
    
    Xtest = X[start_idx:end_idx]
    Ytest = Y[start_idx:end_idx]
    
    Xapp = np.concatenate([X[:start_idx], X[end_idx:]])
    Yapp = np.concatenate([Y[:start_idx], Y[end_idx:]])
    
    return Xapp, Yapp, Xtest, Ytest


# code de la validation croisée (version qui respecte la distribution des classes)

def crossval_strat(X, Y, n_iterations, iteration):
    # Création d'un dictionnaire pour stocker les indices des exemples pour chaque classe
    indices_par_classe = {}
    for label in np.unique(Y):
        indices_par_classe[label] = np.where(Y == label)[0]

    # Initialisation des listes pour stocker les indices d'apprentissage et de test
    indices_apprentissage = []
    indices_test = []

    # Pour chaque classe, divisez les indices en ensembles d'apprentissage et de test
    for indices_classe in indices_par_classe.values():
        taille_classe = len(indices_classe)
        test_size = taille_classe // n_iterations

        start_index = iteration * test_size
        end_index = min((iteration + 1) * test_size, taille_classe)

        indices_apprentissage.extend(indices_classe[:start_index])
        indices_apprentissage.extend(indices_classe[end_index:])
        indices_test.extend(indices_classe[start_index:end_index])

    # Conversion des listes d'indices en tableaux numpy
    indices_apprentissage = np.array(indices_apprentissage)
    indices_test = np.array(indices_test)

    # Séparation des données en ensembles d'apprentissage et de test
    Xapp, Yapp = X[indices_apprentissage], Y[indices_apprentissage]
    Xtest, Ytest = X[indices_test], Y[indices_test]

    return Xapp, Yapp, Xtest, Ytest
    
# ----------------------------------- 
def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    moyenne = np.mean(L)
    ecart_type = np.std(L)
    return moyenne, ecart_type 
    
#------------------------------------
def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    perf = []
    
    for i in range(nb_iter):
        Xapp, Yapp, Xtest, Ytest = crossval_strat(DS[0], DS[1], nb_iter, i)
    
        # Entraînement du perceptron Biais
        perceptron = copy.deepcopy(C)
        perceptron.train(Xapp, Yapp)
        
        # Évaluation de la performance sur l'ensemble de test
        accuracy = perceptron.accuracy(Xtest, Ytest)
        perf.append(accuracy)
        
        print(f"Itération {i} : taille base app.= {len(Xapp)}	taille base test= {len(Xtest)}	Taux de bonne classif:  {accuracy:0.4f}")
    
    taux_moyen, taux_ecart = analyse_perfs(perf)
    
    return (perf, taux_moyen, taux_ecart)  

#---------------------------------------------
def validation_croisee_noprint(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    perf = []
    
    for i in range(nb_iter):
        Xapp, Yapp, Xtest, Ytest = crossval_strat(DS[0], DS[1], nb_iter, i)
    
        # Entraînement du perceptron Biais
        perceptron = copy.deepcopy(C)
        perceptron.train(Xapp, Yapp)
        
        # Évaluation de la performance sur l'ensemble de test
        accuracy = perceptron.accuracy(Xtest, Ytest)
        perf.append(accuracy)
        
    taux_moyen, taux_ecart = analyse_perfs(perf)
    
    return (perf, taux_moyen, taux_ecart) 

