# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2023-2024, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2024

# Import de packages externes
import numpy as np
import pandas as pd
import copy

# ---------------------------
class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        
        
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """ 
        pass

    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        pass
    

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        pass

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        """
        # Appliquer la méthode predict à chaque ligne de desc_set en utilisant np.vectorize
        predictions = map(self.predict, desc_set)

        # Comparer les prédictions avec les labels pour calculer la précision
        correct_predictions = 0
        for i in range(len(desc_set)):
            if self.predict(desc_set[i]) == label_set[i] :
                correct_predictions += 1

        accuracy = correct_predictions / len(label_set)

        return accuracy
        """
        yhat = np.array([self.predict(x) for x in desc_set])
        return np.where(label_set == yhat, 1., 0.).mean()
        
# ------------------------------



class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        super().__init__(input_dimension)
        self.k = k
        
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        #calcul des distances
        distances = [np.linalg.norm(x - y) for y in self.desc_set]
        knn_index= np.argsort(distances)[:self.k]
        
        
        proportion_of_ones = np.sum(self.label_set[knn_index] == 1) / self.k
        return proportion_of_ones
            
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        return 1 if self.score(x)>0.5 else -1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.desc_set = desc_set
        self.label_set = label_set
        

# ------------------------ A COMPLETER : DEFINITION DU CLASSIFIEUR PERCEPTRON


class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    
    def __init__(self, input_dimension, learning_rate=0.01, init=True ):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        super(ClassifierPerceptron,self).__init__(input_dimension)
        self.learning_rate = learning_rate
        if init:
            self.w = np.zeros(input_dimension)
        else:
            self.w = (2*np.random.rand(input_dimension)-1) * 0.001
        self.allw =[self.w.copy()] # stockage des premiers poids 
        #print("w = ", self.w)
        #print("allw = ", self.allw)
            
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        index = [i for i in range(len(desc_set))]
        np.random.shuffle(index)
        
        
        
        for i in index: 
            xi = desc_set[i]
            yi_predict = self.predict(xi)
            yi = label_set[i]
            
            scorex = self.score(xi)
            if yi_predict * yi <= 0 :
                self.w = self.w + self.learning_rate * yi  * xi
                self.allw.append(copy.deepcopy(self.w))

     
    
    def train(self, desc_set, label_set, nb_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """        
        differences = []
        for i in range(nb_max):
            before = np.copy(self.w)
            self.train_step(desc_set, label_set)
            difference = np.linalg.norm(self.w - before)
            differences.append(difference)
            if difference < seuil:
                break
        return differences

    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(self.w, x)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return np.sign(self.score(x))
        
    def get_allw(self):
    	""" rend l'attribut allw
    	"""
    	return self.allw
    
#--------------------------------------


class ClassifierPerceptronBiais(ClassifierPerceptron):
    """ Perceptron de Rosenblatt avec biais
        Variante du perceptron de base
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        # Appel du constructeur de la classe mère
        super().__init__(input_dimension, learning_rate, init)
        # Affichage pour information (décommentez pour la mise au point)
        # print("Init perceptron biais: w= ",self.w," learning rate= ",learning_rate)
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """  
        
        
        index = [i for i in range(len(desc_set))]
        np.random.shuffle(index)
        
        for i in index: 
            xi = desc_set[i]
           #yi_predict = self.predict(xi)
            yi = label_set[i]
            
            scorex = self.score(xi)
            #print(scorex, yi)
            if (scorex*yi < 1) :
                self.w = self.w + self.learning_rate * (yi - scorex) * xi
                self.allw.append(copy.deepcopy(self.w))
                #print(self.allw)
# ------------------------ 

# Vous pouvez avoir besoin d'utiliser la fonction deepcopy de la librairie standard copy:
import copy 


# ------------------------ A COMPLETER :

class ClassifierMultiOAA(Classifier):
    """ Classifieur multi-classes
    """
    def __init__(self, input_dimension, cl_bin):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - cl_bin: classifieur binaire positif/négatif
            Hypothèse : input_dimension > 0
        """
        super().__init__(input_dimension)
        self.cl_bin = cl_bin
        self.classifiers = []  
        
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.classifiers = []  # Réinitialisation des classifieurs
        classes = np.unique(label_set)  # Liste des classes uniques
        for c in classes:
            # Création d'un classifieur binaire pour la classe c
            cl = copy.deepcopy(self.cl_bin)
            # Redéfinition des étiquettes pour la classe c
            ytmp = np.where(label_set == c, 1, -1)
            # Entraînement du classifieur binaire pour la classe c
            cl.train(desc_set, ytmp)
            # Ajout du classifieur à la liste
            self.classifiers.append(cl)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        scores = []
        for cl in self.classifiers :
            scores.append(cl.score(x))
        return np.array(scores)
        
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        scores = self.score(x)
        return np.argmax(scores)
        

   
        
