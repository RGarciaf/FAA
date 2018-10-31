from Datos import Datos
from EstrategiaParticionado import *
from Clasificador import *
import numpy as np
from os import walk, getcwd, path
import re

class Roc(object):
    def __init__(self):
        roc = []
        estrategia = ValidacionSimple()
        clas = ClasificadorNaiveBayes()
        dataset = Datos("ConjuntosDatos/balloons.data")
        roc.append(clas.roc(estrategia,dataset,clas))

        dataset = Datos("ConjuntosDatos/german.data")
        estrategia = ValidacionSimple()
        clas = ClasificadorNaiveBayes()
        roc.append(clas.roc(estrategia,dataset,clas))

        dataset = Datos("ConjuntosDatos/tic-tac-toe.data")
        estrategia = ValidacionSimple()
        clas = ClasificadorNaiveBayes()
        roc.append(clas.roc(estrategia,dataset,clas))
        self.roc = roc
