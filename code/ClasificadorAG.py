from abc import ABCMeta,abstractmethod
from multipledispatch import dispatch
import random
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



class ClasificadorAG(Clasificador):
    
    def __init__(self, n_cromosomas, dataset, regla_entera = True):
        self.cromosomas = []
        self.generarPoblacion(n_cromosomas, dataset, regla_entera)
    
    def ordenarIndividuos(self):
        return sorted(self.cromosomas)
    
    def generarPoblacion(self, n_cromosomas, dataset, regla_entera):
        for _ in range(n_cromosomas):
            self.cromosomas.append(self.Cromosoma(dataset, regla_entera))
    
    def recombinar(self):
        poblacion = []
        for cromosoma in self.cromosomas:
            poblacion.append(cromosoma.recombinar(random.sample(self.cromosomas, 1)))
    
    def mutar(self):
        poblacion = []
        for cromosoma in self.cromosomas:
            poblacion.append(cromosoma.mutar()
    
    
    class Cromosoma:
        
        def __init__(self, dataset, regla_entera = True):
            self.regla_entera = regla_entera
            self.n_attrs = len(dataset.nombreAtributos) - 1  #nº attrs menos la clase
            self.rlen = randint(1,pow(dataset.k, self.n_attrs))
            
            self.n_intervalos = dataset.k
            
            self.reglas = set()
            
            
            self.fit = -1
        def generarReglas(self,):
            for _ in range(rlen):
                regla = self.crearRegla(dataset, self.regla_entera) 
                
                while regla in self.reglas:
                    regla = self.crearRegla(dataset, regla_entera)
                    
                self.reglas.add(regla)
        
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
            if not datos:
                return self.fit
                
            for fila in datos:
                self.fit += self.compruebaReglas(fila)
            self.fit /= len(datos)
                
            return self.fit
            
        def compruebaReglas(self, fila):
            for regla in self.reglas:
                if self.compruebaRegla(fila, regla) == 1:
                    return 1
            return 0;
        
        def compruebaRegla(self, fila, regla):
            for dato_fila, dato_regla in zip(fila, regla):
                if dato_regla != 0 and dato_regla != dato_fila:
                    return 0
                    
            if fila[-1] == regla[-1]:
                return 1
            return 0
        
        def __lt__(self, other):
            return self.fitness() < other.fitness()
            
        def recombinar(self, cromosoma):
            return self.recombina_reglas(cromosoma)
        
        def recombinar_regla(self, cromosoma):
            new_cromosoma = 
            for regla in self.reglas:
                alea = tirarDado(len(regla)-1)
                regla[:alea] + regla[alea:]
        
        def recombinar_reglas(self, cromosoma):
            for regla in self.reglas:
                alea = tirarDado(len(regla)-1)
            return regla[:alea] + regla[alea:]
        
        def recombinar_cromosomas(self, cromosoma):
            for regla in self.reglas:
                alea = tirarDado(len(regla)-1)
            return regla[:alea] + regla[alea:]
        
        def mutar(self, porcentaje = 4):
            for regla in self.reglas:
                if tirarDado(99) < porcentaje:
                    for indice, intervalo in enumerate(regla[:-1]):
                        if tirarDado(99) > porcentaje:
                            regla[indice] = tirarDado(self.n_intervalos)
        
                            
def tirarDado(caras, ini = 0):
    return random.randint(ini,caras)