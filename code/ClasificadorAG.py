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
    
    def __init__(self, n_cromosomas, dataset, n_generaciones, regla_entera = True):
        self.n_generaciones = n_generaciones
        self.cromosomas = []
        self.datosIntervalizados = []
        self.n_cromosomas = n_cromosomas
        self.generarPoblacion(n_cromosomas, dataset, regla_entera)
        self.hulk = None
        
        
    @staticmethod
    def ruleta_rusa(cromosomas, n_individuos):
        seleccion = np.array()
        total_fitness = np.sum(map(lambda x:x.fitness(),cromosomas))
        for cromosoma in cromosomas:
            seleccion = np.append(seleccion, 
                            np.tile(cromosoma, 
                                    int((cromosoma.fitness()/total_fitness) * n_individuos)))
        
        return seleccion
        
    
    def ordenarCromosomas(self):
        return sorted(self.cromosomas)
    
    def generarPoblacion(self, n_cromosomas, dataset, regla_entera):
        a, k = dataset.crearIntervalos(dataset.datos)
        self.datosIntervalizados = dataset.convertirAIntervalos(dataset.datos)
        for _ in range(n_cromosomas):
            self.cromosomas.append(self.Cromosoma(dataset, regla_entera = regla_entera))
    
    def recombinar(self):
        poblacion = []
        ruleta = ClasificadorAG.ruleta_rusa(self.cromosomas,len(self.cromosomas))
        for cromosoma in self.cromosomas:
            poblacion.append(cromosoma.recombinar(ruleta.pop(random.randint(0, len (ruleta)))))
        return poblacion
    
    def mutar(self):
        poblacion = []
        for cromosoma in self.cromosomas:
            poblacion.append(cromosoma.mutar())
        return poblacion
    
    def elitismo(self, porcentaje = 2):
        indices = int((len(self.cromosomas) * porcentaje) / 100)
        return self.cromosomas[:indices]
            
    def entrenar(self, datos):
        for _ in range(self.n_generaciones):
            self.cromosomas = self.ordenarCromosomas()[:self.n_cromosomas] 
            recombinado = self.recombinar()     
            mutados = self.mutar()              
            self.cromosomas = self.elitismo() + recombinado + mutados 
        
        self.hulk = self.ordenarCromosomas()[0]
    
    def clasifica(self, datos):
        return self.hulk.clasifica(datos)
    
    class Cromosoma:
        
        def __init__(self, dataset, reglas = None, regla_entera = True):
            self.regla_entera = regla_entera
            self.n_attrs = len(dataset.nombreAtributos) - 1  #nº attrs menos la clase
            #self.rlen = randint(1, pow(dataset.k, self.n_attrs))
            self.rlen = randint(2, 20)
            self.datos = dataset.datos
            self.n_intervalos = dataset.k
            print("hola reglas", 
            if reglas:
                self.reglas = reglas
            else:
                self.reglas = set()
                self.generarReglas(dataset)
            
            self.fit = -1
            
        def generarReglas(self, dataset):
            regla = None
            for _ in range(self.rlen):
                regla = self.Regla(dataset, self.regla_entera)
                
                while regla in self.reglas:
                    regla = self.Regla(dataset, self.regla_entera)
                    
                self.reglas.add(regla)
        
        def fitness(self, datos = None):
            if not datos and self.fit > -1:
                return self.fit
            
            if not datos:
                datos = self.datos
                
            for fila in datos:
                self.fit += self.compruebaReglas(fila)
            self.fit /= len(datos)
                
            return self.fit
            
        def compruebaReglas(self, fila):
            for regla in self.reglas:
                if regla.compruebaRegla(fila) == 1:
                    return 1
            return 0;
            
        def clasifica(self, datos):
            clase = {0:0,1:0}
            for fila in datos:
                if self.compruebaReglas(fila) == 1:
                    clase[fila[-1]] += 1
            return clase;
        
        def __lt__(self, other):
            return self.fitness() < other.fitness()
            
        def recombinar(self, cromosoma):
            if tirarDado(99) < 80:
                return self.recombinar_cromosomas(cromosoma)
            return self
        
        # def recombinar_regla(self, cromosoma):
        #     new_cromosoma = 
        #     for regla in self.reglas:
        #         alea = tirarDado(len(regla)-1)
        #         regla[:alea] + regla[alea:]
        
        # def recombinar_reglas(self, cromosoma):
        #     for regla in self.reglas:
        #         alea = tirarDado(len(regla)-1)
        #     return regla[:alea] + regla[alea:]
        
        def recombinar_cromosomas(self, cromosoma):
            medio = round(len(self.reglas)/2)
            medio_other = round(len(cromosoma.reglas)/2)
            
            return Cromosoma(self.dataset, reglas = self.reglas[:medio] + cromosoma.reglas[:medio_other], regla_entera = self.regla_entera )
        
        def mutar(self, porcentaje = 4):
            for regla in self.reglas:
                if tirarDado(99) < porcentaje:
                    regla.mutar(porcentaje)
        
        class Regla():
            def __init__(self,dataset, regla_entera = True):
                self.regla_entera = regla_entera
                self.valores = []
                self.n_intervalos = dataset.k  #nº attrs menos la clase
                print("hola")
                
                if regla_entera:
                    print("hola1")
                    self.valores = np.append(np.random.randint(dataset.k + 1, size = len(dataset.nombreAtributos)-1), np.random.randint(len(dataset.diccionarios['Class']), size = 1))
                    print("regla entera", self.valores)
                else:
                    print("hola2")
                    regla = []
                    for _ in range(len(dataset.nombreAtributos)-1):
                        
                        regla.append(np.random.randint(1, size = dataset.k))
                    regla.append(np.random.randint(len(dataset.diccionarios['Class']), size = 1))
                    self.valores = regla
                    print("regla binaria", self.valores)
                    
            def compruebaRegla(self, fila):
                if self.regla_entera:
                    for dato_fila, dato_regla in zip(fila[:-1], self.valores[:-1]):
                        if dato_regla != 0 and dato_regla != dato_fila:
                            return 0
                else:
                    for dato_fila, dato_regla in zip(fila[:-1], self.valores[:-1]):
                        if np.sum(dato_regla) > 0:
                            if self.compruebaAtributo(dato_fila,dato_regla) == 0:
                                return 0
                            
                if fila[-1] == self.valores[-1]:
                    return 1
                return 0
                    
            def compruebaAtributo(self, dato_fila, intervalos):
                for index, intervalo in enumerate(intervalos):
                    if intervalo == 1 and index +1 == dato_fila:
                        return 1
                return 0
                
            def mutar(self, porcentaje = 4):
                if self.regla_entera:
                    for indice, intervalo in enumerate(self.valores[:-1]):
                        if tirarDado(99) > porcentaje:
                            self.valores[indice] = tirarDado(self.n_intervalos)
                else: 
                    for index_attr, attr in enumerate(self.valores[:-1]):
                        for index_intervalo, intervalo in enumerate(attr):
                            if tirarDado(99) > porcentaje:
                                self.valores[index_attr][index_intervalo] = tirarDado(1)
                    
def tirarDado(caras, ini = 0):
    return random.randint(ini,caras)
    
    
