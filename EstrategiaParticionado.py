from abc import ABCMeta,abstractmethod
import numpy as np
import random

class Particion():

  # Esta clase mantiene la lista de �ndices de Train y Test para cada partici�n del conjunto de particiones
  def __init__(self):
    self.indicesTrain=[]
    self.indicesTest=[]

#####################################################################################################

class EstrategiaParticionado(object):

  # Clase abstracta
  __metaclass__ = ABCMeta


  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta: nombreEstrategia, numeroParticiones, listaParticiones. Se pasan en el constructor

  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada estrategia concreta
  def creaParticiones(self,datos,seed=None):
    pass


#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):

    # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
    # Devuelve una lista de particiones (clase Particion)
    # TODO: implementar

    def creaParticiones(self,datos,seed=None):
        par = Particion()
        k = int(round(self.porcentaje*len(datos.datos)))
        perms = np.arange(len(datos.datos))
        permutacion = np.random.permutation(perms)
        par.indicesTrain = (np.append(par.indicesTrain, permutacion[:k])).astype(int)
        par.indicesTest = (np.append(par.indicesTest, permutacion[k:])).astype(int)
        return par


    def __init__(self, nombreEstrategia, numeroParticiones, porcentaje, dataset):
        self.nombreEstrategia= nombreEstrategia
        self.numeroParticiones = numeroParticiones
        self.porcentaje = porcentaje
        self.particiones = {}
        for particion in range(numeroParticiones):
            self.particiones[particion] = self.creaParticiones(dataset)

    


#####################################################################################################
class ValidacionCruzada(EstrategiaParticionado):

  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  # TODO: implementar
    def creaParticiones(self,datos,seed=None):
        particiones = {}
        perms = np.arange(len(datos.datos))
        permutacion = np.random.permutation(perms)

        for i in range(0,len(datos.datos), self.numeroParticiones):
            par = Particion()
            par.indicesTest = np.append([],permutacion[i:i+self.numeroParticiones]).astype(int)
            par.indicesTrain = np.append(permutacion[:i], permutacion[i+self.numeroParticiones:]).astype(int)
            particiones[i] = par

        return particiones

    
    def __init__(self, nombreEstrategia, numeroParticiones, dataset):
        self.nombreEstrategia = nombreEstrategia
        self.numeroParticiones = numeroParticiones
        self.particiones = {}
        self.particiones = self.creaParticiones(dataset)

  

    #####################################################################################################
class ValidacionBootstrap(EstrategiaParticionado):
    
  # Crea particiones segun el metodo de validacion por bootstrap.
  # Esta funcion devuelve una lista de particiones (clase Particion)
  
  # TODO: implementar
      
    def creaParticiones(self, datos,seed=None):
        particiones = {}
        perms = np.arange(len(datos.datos))
        for i in range(self.numeroParticiones):
            par = Particion()
            for _ in range(self.tamParticion):
                par.indicesTest = np.append(par.indicesTest,random.choice(perms)).astype(int)
            par.indicesTrain = np.setdiff1d(perms, par.indicesTest)
            particiones[i] = par
        return particiones
    
    
    def __init__(self, nombreEstrategia, numeroParticiones, tamParticion, dataset):
        self.nombreEstrategia = nombreEstrategia
        self.numeroParticiones = numeroParticiones
        self.tamParticion = tamParticion
        self.particiones = {}
        self.particiones = self.creaParticiones(dataset)

  
