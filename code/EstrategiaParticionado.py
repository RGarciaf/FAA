from abc import ABCMeta,abstractmethod
<<<<<<< HEAD:Plantillas/EstrategiaParticionado.py

=======
import numpy as np
>>>>>>> ccd947ceaf0edbe860ff894d1c83aacb534d6559:code/EstrategiaParticionado.py

class Particion():

  # Esta clase mantiene la lista de �ndices de Train y Test para cada partici�n del conjunto de particiones
  def __init__(self):
    self.indicesTrain=[]
    self.indicesTest=[]

#####################################################################################################

class EstrategiaParticionado(object):

  # Clase abstracta
  __metaclass__ = ABCMeta
<<<<<<< HEAD:Plantillas/EstrategiaParticionado.py
  numeroParticiones
=======
>>>>>>> ccd947ceaf0edbe860ff894d1c83aacb534d6559:code/EstrategiaParticionado.py


  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta: nombreEstrategia, numeroParticiones, listaParticiones. Se pasan en el constructor

  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada estrategia concreta
  def creaParticiones(self,datos,seed=None):
    pass


#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):

<<<<<<< HEAD:Plantillas/EstrategiaParticionado.py
  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
  # Devuelve una lista de particiones (clase Particion)
  # TODO: implementar

  def __init__(self, nombreEstrategia, numeroParticiones, listaParticiones, porcentaje):
      self.nombreEstrategia = nombreEstrategia
      self.numeroParticiones = numeroParticiones
      self.listaParticiones = listaParticiones
      self.porcentaje = porcentaje


  def creaParticiones(self,datos,seed=None):
    par = Particion()
    perms = np.indices(self.numeroParticiones,len(datos))
    for ar in perms:
        permutacion = np.random.permutation(ar)
        par.indicesTrain.append(permutacion[:self.porcentaje*len(datos)])
        par.indicesTest.append(permutacion[self.porcentaje*len(datos):])
    return par
=======
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

    
>>>>>>> ccd947ceaf0edbe860ff894d1c83aacb534d6559:code/EstrategiaParticionado.py


#####################################################################################################
class ValidacionCruzada(EstrategiaParticionado):

  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  # TODO: implementar
<<<<<<< HEAD:Plantillas/EstrategiaParticionado.py

  def __init__(self, nombreEstrategia, numeroParticiones, listaParticiones):
      self.nombreEstrategia = nombreEstrategia
      self.numeroParticiones = numeroParticiones
      self.listaParticiones = listaParticiones

  def creaParticiones(self,datos,seed=None):
    par = Particion()
    perms = np.arange(len(datos))
    for i in range(0,len(datos)/self.numeroParticiones, self.numeroParticiones):
        par.indicesTest.append(perms[i:i+self.numeroParticiones])
        par.indicesTrain.append(perms[:i] + perms[i+self.numeroParticiones:])

    return par


#####################################################################################################
=======
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
>>>>>>> ccd947ceaf0edbe860ff894d1c83aacb534d6559:code/EstrategiaParticionado.py
class ValidacionBootstrap(EstrategiaParticionado):

  # Crea particiones segun el metodo de validacion por bootstrap.
  # Esta funcion devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def __init__(self, nombreEstrategia, numeroParticiones, listaParticiones, numTablas):
      self.nombreEstrategia = nombreEstrategia
      self.numeroParticiones = numeroParticiones
      self.listaParticiones = listaParticiones
      self.numTablas = numTablas

  def creaParticiones(self,datos,seed=None):
    par = Particion()

    for i in range(self.numTablas):
        perms = np.arange(len(datos))
        test = []
        for j in range(self.numeroParticiones):
            np.random.shuffle(perms)
            test.append(perms[j])
        par.indicesTest.append(test)
        par.indicesTrain.append(list(set(perms) - set(test)))

    return par
