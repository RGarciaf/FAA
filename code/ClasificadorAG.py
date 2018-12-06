from abc import ABCMeta,abstractmethod
import numpy as np
from random import randint
from scipy.stats import norm
import scipy
import pprint

class Clasificador(object):

    # Clase abstracta
    __metaclass__ = ABCMeta

    # Metodos abstractos que se implementan en casa clasificador concreto
    @abstractmethod
    # TODO: esta funcion deben ser implementadas en cada clasificador concreto
    # datosTrain: matriz numpy con los datos de entrenamiento
    # atributosDiscretos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
    def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
        pass


    @abstractmethod
    # TODO: esta funcion deben ser implementadas en cada clasificador concreto
    # devuelve un numpy array con las predicciones
    def clasifica(self,datosTest,atributosDiscretos,diccionario):
        pass



    # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
    def error(self,datos,pred):
    #     # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
        clases = np.column_stack(datos)[-1]
        error = 0
        for clase, clase_pred in zip(clases, pred):
            if clase != clase_pred:
                error += 1

        error = error/len(datos)
        return error



    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    # TODO: implementar esta funcion
    def validacion(self,particionado,dataset,clasificador,seed=None):
        errores = []
        particionado.creaParticiones(dataset)
        for particion in particionado.particiones:
            clasificador.entrenamiento(dataset.extraeDatos(particion.indicesTrain), dataset.tipoAtributos, dataset.diccionarios)
            clasificacion = clasificador.clasifica(dataset.extraeDatos(particion.indicesTest), dataset.tipoAtributos, dataset.diccionarios)
            errores.append(clasificador.error(dataset.extraeDatos(particion.indicesTest), clasificacion))
        return errores

    @abstractmethod
    def roc(self,particionado,dataset,clasificador):
        pass

class ClasificadorAG():
    
    def __init__(self, n_cromosomas, dataset, regla_entera = True):
        self.cromosomas = []
        self.generarPoblacion(n_cromosomas, dataset, regla_entera)
    
    def ordenarIndividuos(self):
        return sorted(self.cromosomas)
    
    def generarPoblacion(self, n_cromosomas, dataset, regla_entera):
        for _ in range(n_cromosomas):
            self.cromosomas.append(self.Cromosoma(dataset, regla_entera))
    
    def recombinar(self):
        pass
    
    def mutar(self):
        pass
    
    
    class Cromosoma():
        
        def __init__(self, dataset, regla_entera):
            n_attrs = len(dataset.nombreAtributos) - 1  #nº attrs menos la clase
            rlen = randint(1,pow(dataset.k, n_attrs))
            
            self.reglas = set()
            for _ in range(rlen):
                regla = self.crearRegla(dataset, regla_entera) 
                
                while regla in self.reglas:
                    regla = self.crearRegla(dataset, regla_entera)
                    
                self.reglas.add(regla)
            
            self.fit = -1
            
        
        def crearRegla(self, dataset, regla_entera):
            if regla_entera:
                return np.append(np.random.randint(dataset.k + 1, size = len(dataset.nombreAtributos)-1), np.random.randint(len(dataset.diccionarios['Class']), size = 1))
            else:
                regla = []
                for _ in range(len(dataset.nombreAtributos)-1):
                    regla.append(np.random.randint(1, size = dataset.k))
                regla.append(np.random.randint(len(dataset.diccionarios['Class']), size = 1))
                return regla
        
        def fitness(self, datos = None):
            if self.fit == -1 or not datos:
                pass
            return self.fitness
        
        def __sizeof__(self):
            return self.fitness()
            
        def recombinar(self, cromosoma):
            pass
        
        def mutar(self):
            pass