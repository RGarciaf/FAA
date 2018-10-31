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
        self.medias = []
    
    def medias(roc):
        TPR = {}
        FPR = {}
        medias = []
        ini = {}
        for clas in roc[0][0]:
            ini.update({clas:0})

        for archivo in roc:
            TPR = ini.copy()
            FPR = ini.copy()
            for particion in archivo:
                for clas in particion:
                    TPR[clas] += particion[clas]["TPR"]
                    FPR[clas] += particion[clas]["FPR"]
            for clas in TPR:
                TPR.update({clas:round(TPR[clas]/len(roc[0]),3)})
                FPR.update({clas:round(FPR[clas]/len(roc[0]),3)})
            medias.append({"TPR":TPR,"FPR":FPR})
        pprint.pprint(medias)
        return medias

    def calcula_medias_roc(self):
        medias = []
        print("\nValidacion simple")
        medias.append(Roc.medias(self.simple))
        print("\nValidacion cruzada")
        medias.append(Roc.medias(self.cruzada))
        print("\nValidacion Bootstrap")
        medias.append(Roc.medias(self.bootstrap))
        self.medias = medias
        return medias

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
