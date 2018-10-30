from Datos import Datos
from EstrategiaParticionado import *
from Clasificador import *
import numpy as np
from os import walk, getcwd, path
import re

class Roc(object):
    def __init__(self, estrategia = ValidacionSimple(), clas = ClasificadorNaiveBayes(True)):
        roc = []
        dataset = Datos("ConjuntosDatos/balloons.data")
        print("Estrategia ", estrategia.nombreEstrategia)
        roc.append(clas.roc(estrategia,dataset,clas))

        dataset = Datos("ConjuntosDatos/german.data")
        print("Estrategia ", estrategia.nombreEstrategia)
        roc.append(clas.roc(estrategia,dataset,clas))

        dataset = Datos("ConjuntosDatos/tic-tac-toe.data")
        print("Estrategia ", estrategia.nombreEstrategia, " ConjuntosDatos/tic-tac-toe.data")
        roc.append(clas.roc(estrategia,dataset,clas))
        self.roc = roc
