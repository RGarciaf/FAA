from abc import ABCMeta,abstractmethod
import numpy as np
import random

class Particion():

  # Esta clase mantiene la lista de indices de Train y Test para cada particion del conjunto de particiones
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


        for i in range(self.numeroParticiones):
            par = Particion()
            k = int(round(self.porcentaje*len(datos.datos)))
            perms = np.arange(len(datos.datos))
            permutacion = np.random.permutation(perms)
            par.indicesTrain = (np.append(par.indicesTrain, permutacion[:k])).astype(int)
            par.indicesTest = (np.append(par.indicesTest, permutacion[k:])).astype(int)
            self.particiones.append(par)
        return self.particiones


    def __init__(self, numeroParticiones = 5, porcentaje = 0.6):
        self.nombreEstrategia= "Validacion Simple"
        self.numeroParticiones = numeroParticiones
        self.porcentaje = porcentaje
        self.particiones = []





#####################################################################################################
class ValidacionCruzada(EstrategiaParticionado):

  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  # TODO: implementar
    def creaParticiones(self,datos,seed=None):
        perms = np.arange(len(datos.datos))
        permutacion = np.random.permutation(perms)

        for i in range(0,len(datos.datos), self.numeroParticiones):
            par = Particion()
            par.indicesTest = np.append([],permutacion[i:i+self.numeroParticiones]).astype(int)
            par.indicesTrain = np.append(permutacion[:i], permutacion[i+self.numeroParticiones:]).astype(int)
            self.particiones.append(par)

        return self.particiones


    def __init__(self, numeroParticiones = 5):
        self.nombreEstrategia = "Validacion Cruzada"
        self.numeroParticiones = numeroParticiones
        self.particiones = []



    #####################################################################################################
class ValidacionBootstrap(EstrategiaParticionado):

  # Crea particiones segun el metodo de validacion por bootstrap.
  # Esta funcion devuelve una lista de particiones (clase Particion)

  # TODO: implementar

    def creaParticiones(self, datos,seed=None):
        particiones = []
        perms = np.arange(len(datos.datos))
        for i in range(self.numeroParticiones):
            par = Particion()
            for _ in range(self.tamParticion):
                par.indicesTest = np.append(par.indicesTest,random.choice(perms)).astype(int)
            par.indicesTrain = np.setdiff1d(perms, par.indicesTest)
            self.particiones.append(par)
        return self.particiones


    def __init__(self, numeroParticiones = 5, tamParticion = 2):
        self.nombreEstrategia = "Validacion Bootstrap"
        self.numeroParticiones = numeroParticiones
        self.tamParticion = tamParticion
        self.particiones = []
