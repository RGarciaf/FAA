from abc import ABCMeta,abstractmethod
from multipledispatch import dispatch
import random
import numpy as np
from random import randint
from scipy.stats import norm
import scipy
import pprint
import copy

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
            clasificador.entrenar(dataset.extraeDatosIntervalos(particion.indicesTrain))
            clasificacion = ( clasificador.clasifica(dataset.extraeDatosIntervalos(particion.indicesTest)))
            errores.append(clasificador.error(dataset.extraeDatosIntervalos(particion.indicesTest), clasificacion))
        return errores

    @abstractmethod
    def roc(self,particionado,dataset,clasificador):
        pass



class ClasificadorAG(Clasificador):
    
    def __init__(self, n_cromosomas, dataset, n_generaciones, regla_entera = True):
        self.n_generaciones = n_generaciones
        self.cromosomas = []
        #self.datosIntervalizados = []
        self.n_cromosomas = n_cromosomas
        self.generarPoblacion(n_cromosomas, dataset, regla_entera)
        self.hulk = None
        self.hulk_gen = []
        self.media_gen = []
        
        
    @staticmethod
    def ruleta_rusa(cromosomas, n_individuos):
        seleccion, probabilidades = [], []

        total_fitness = 0
        for cromosoma in cromosomas:
            total_fitness += cromosoma.fitness()
            
        # print(total_fitness)

        if total_fitness == 0:
            return cromosomas
        for cromosoma in cromosomas:
            probabilidades.append(cromosoma.fitness()/total_fitness)
        seleccion = np.random.choice(cromosomas, n_individuos, p=probabilidades)
        return seleccion.tolist()
        
    
    def ordenarCromosomas(self, datos = np.array([])): #Hace falta que sea array aunque este vacio
        if datos.any(): #Si existe algun elmento, se mete (si hay datos)
            for cromosoma in self.cromosomas:
                cromosoma.fitness(datos)
        return sorted(self.cromosomas)
    
    def generarPoblacion(self, n_cromosomas, dataset, regla_entera):
        for _ in range(n_cromosomas):
            self.cromosomas.append(self.Cromosoma(dataset = dataset, regla_entera=regla_entera))

    def recombinar(self):
        poblacion = []
        ruleta = ClasificadorAG.ruleta_rusa(self.cromosomas,len(self.cromosomas))
        for cromosoma in self.cromosomas:
            poblacion.append(cromosoma.recombinar(ruleta.pop(random.randint(0, len (ruleta) - 1))))
        return poblacion
    
    def mutar(self):
        poblacion = []
        for cromosoma in self.cromosomas:
            cromosoma.mutar()

    def elitismo(self, porcentaje = 2):
        indices = int((len(self.cromosomas) * porcentaje) / 100)
        return [ self.Cromosoma(el.dataset, el.reglas, el.regla_entera) for el in self.cromosomas[:indices]] 
        
            
    def entrenar(self, datos =  np.array([])):
        for i in range(self.n_generaciones):
            print("\tGeneracion ", i)           
            self.cromosomas = self.ordenarCromosomas(datos)[:self.n_cromosomas] 
            self.hulk_gen.append(self.cromosomas[0])
            self.media_gen.append(np.sum([el.fitness(datos) for el in self.cromosomas])/len(self.cromosomas)) 
            #print("cromosomas:")
            #print([el.fitness(datos) for el in self.cromosomas])

            elite = self.elitismo()            
            #print("elite:")
            #print([el.fitness(datos) for el in elite])
            
            recombinado = self.recombinar()     
            self.mutar()              
            self.cromosomas = elite + recombinado 
            # for el in self.cromosomas:
            #     el.fit = -1
            #print("elite:")
            #print([el.fitness(datos) for el in elite])
            #print(len(self.cromosomas), "\n")
            
            
        self.hulk = self.ordenarCromosomas()[0]
        return self.hulk
    
    def clasifica(self, datos):
        return self.hulk.clasifica(datos)
    
    class Cromosoma:
        
        def __init__(self, dataset, reglas = None, regla_entera = True):
            self.regla_entera = regla_entera
            self.n_attrs = len(dataset.nombreAtributos) - 1  #nº attrs menos la clase
            # self.rlen = pow(dataset.k, self.n_attrs)
            # self.rlen = randint(1, pow(dataset.k, self.n_attrs))
            self.rlen = randint(1, 100)
            self.datos = dataset.convertirAIntervalos(dataset.datos)
            self.n_intervalos = dataset.k
            self.dataset = dataset
            self.prior = 0
            
            if reglas != None:
                self.reglas = set()
                for regla in reglas:
                    self.reglas.add(self.Regla(dataset,regla_entera,regla.valores))
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
        
        def fitness(self, datos = np.array([])):
            if not datos.any():     #Si no se cumple que alguno de los datos exista, porque no hay ninguno, se mete
                if self.fit > -1:   
                    return self.fit
                else:
                    datos = self.datos            
            
            self.fit = 0
            for fila in datos:
                self.fit += self.compruebaReglas(fila)
            self.fit /= len(datos)
            c, counts = np.unique(datos[-1], return_counts=True)
            if counts[0] < counts[1]:
                self.prior = 1
            else:
                self.prior = 0
            
            return self.fit
            
        def compruebaReglas(self, fila):
            if self.clasificaReglas(fila) == fila[-1]:
                return 1
            return 0

        def clasificaReglas(self, fila):
            aciertos = 0
            nreglas = 0
            for regla in self.reglas:
                if regla.clasifica(fila) == 1:
                    aciertos += regla.valores[-1]
                    nreglas += 1
            if nreglas > 0:
                aciertos /= nreglas
                return round(aciertos)
            return self.prior
            
            
        def clasifica(self, datos):
            clase = []
            for fila in datos:
                # print(fila[-1])                   
                clase.append(self.clasificaReglas(fila))
            return clase
        
        def __lt__(self, other):
            # print("Estoy en el lt")
            # print(self.fitness(), " > ", other.fitness(), self.fitness() > other.fitness() )
            return self.fitness() > other.fitness()

        def __gt__(self, other):
            # print("Estoy en el gt")
            # print(self.fitness(), " < ", other.fitness(), self.fitness() < other.fitness() )
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
            
            return ClasificadorAG.Cromosoma(self.dataset, reglas = set(list(self.reglas)[:medio] + list(cromosoma.reglas)[:medio_other]), regla_entera = self.regla_entera )
        
        def mutar(self, porcentaje = 4):
            for regla in self.reglas:
                if tirarDado(99) < porcentaje:
                    regla.mutar(porcentaje)
            
        
        class Regla():
            def __init__(self,dataset, regla_entera = True, valores = np.array([])):
                self.regla_entera = regla_entera
                self.valores = []
                if np.array(valores).any():
                    self.valores = copy.deepcopy(valores)
                else:
                    if regla_entera:
                        self.valores = np.append(np.random.randint(dataset.k + 1, size = len(dataset.nombreAtributos)-1), np.random.randint(len(dataset.diccionarios['Class']), size = 1))
                    else:
                        regla = []
                        for _ in range(len(dataset.nombreAtributos)-1):
                            regla.append(np.random.randint(2, size = dataset.k).tolist())
                        regla.append(np.random.randint(len(dataset.diccionarios['Class']), size = 1)[0])
                        self.valores = regla
                self.n_intervalos = dataset.k  #nº attrs menos la clase
                
                
                    
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

            def clasifica(self, fila):
                if self.regla_entera:
                    for dato_fila, dato_regla in zip(fila[:-1], self.valores[:-1]):
                        if dato_regla != 0 and dato_regla != dato_fila:
                            return 0
                else:
                    for dato_fila, dato_regla in zip(fila[:-1], self.valores[:-1]):
                        if np.sum(dato_regla) > 0:
                            if self.compruebaAtributo(dato_fila,dato_regla) == 0:
                                return 0                            
                return 1
                    
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