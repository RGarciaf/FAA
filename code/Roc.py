from Datos import Datos
from EstrategiaParticionado import *
from Clasificador import *
import numpy as np
from os import walk, getcwd, path
import re

class Roc(object):
    def __init__(self):
        self.simple = []
        self.cruzada = []
        self.bootstrap = []
    
    def medias_roc(self):
        balloons = Datos("ConjuntosDatos/balloons.data")
        german = Datos("ConjuntosDatos/german.data")
        tic_tac_toe = Datos("ConjuntosDatos/tic-tac-toe.data")
        
        print("Validacion simple")
        simple = []
        estrategia = ValidacionSimple()
        clas = ClasificadorNaiveBayes()        
        print("\tBalloons")
        simple.append(clas.roc(estrategia,balloons,clas))

        print("\tGerman")
        estrategia = ValidacionSimple()
        clas = ClasificadorNaiveBayes()
        simple.append(clas.roc(estrategia,german,clas))

        print("\tTic-tac-toe")
        estrategia = ValidacionSimple()
        clas = ClasificadorNaiveBayes()
        simple.append(clas.roc(estrategia,tic_tac_toe,clas))
        self.simple = simple

        print("Validacion cruzada")
        cruzada = []
        estrategia = ValidacionCruzada()
        clas = ClasificadorNaiveBayes()
        print("\tBalloons")
        cruzada.append(clas.roc(estrategia,balloons,clas))

        estrategia = ValidacionCruzada()
        clas = ClasificadorNaiveBayes()
        print("\tGerman")
        cruzada.append(clas.roc(estrategia,german,clas))

        estrategia = ValidacionCruzada()
        clas = ClasificadorNaiveBayes()
        print("\tTic-tac-toe")
        cruzada.append(clas.roc(estrategia,tic_tac_toe,clas))
        self.cruzada = cruzada

        print("Validacion Bootstrap")
        bootstrap = []
        estrategia = ValidacionBootstrap()
        clas = ClasificadorNaiveBayes()   
        print("\tBalloons")     
        bootstrap.append(clas.roc(estrategia,balloons,clas))

        estrategia = ValidacionBootstrap()
        clas = ClasificadorNaiveBayes()    
        print("\tGerman")    
        bootstrap.append(clas.roc(estrategia,german,clas))

        estrategia = ValidacionBootstrap()
        clas = ClasificadorNaiveBayes()  
        print("\tTic-tac-toe")      
        bootstrap.append(clas.roc(estrategia,tic_tac_toe,clas))
        self.bootstrap = bootstrap

        return simple, cruzada, bootstrap








        # estrategia = ValidacionSimple()
        # clas = ClasificadorNaiveBayes()
        # dataset = Datos("ConjuntosDatos/balloons.data")
        # roc.append(clas.roc(estrategia,dataset,clas))

        # dataset = Datos("ConjuntosDatos/german.data")
        # estrategia = ValidacionSimple()
        # clas = ClasificadorNaiveBayes()
        # roc.append(clas.roc(estrategia,dataset,clas))

        # dataset = Datos("ConjuntosDatos/tic-tac-toe.data")
        # estrategia = ValidacionSimple()
        # clas = ClasificadorNaiveBayes()
        # roc.append(clas.roc(estrategia,dataset,clas))
        # self.roc = roc
