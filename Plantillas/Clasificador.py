from abc import ABCMeta,abstractmethod


class Clasificador(object):

  # Clase abstracta
  __metaclass__ = ABCMeta

  # Metodos abstractos que se implementan en casa clasificador concreto
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada clasificador concreto
  # datosTrain: matriz numpy con los datos de entrenamiento
  # atributosDiscretos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion
  # de variables discretas
  def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
    pass


  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada clasificador concreto
  # devuelve un numpy array con las predicciones
  def clasifica(self,datosTest,atributosDiscretos,diccionario):
    pass


  # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
  # TODO: implementar
  def error(self,datos,pred):
    # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
	pass


  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  # TODO: implementar esta funcion
  def validacion(self,particionado,dataset,clasificador,seed=None):

    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test
	pass


##############################################################################

class ClasificadorNaiveBayes(Clasificador):


    Clasificador.entrenamiento(self,datostrain,atributosDiscretos,diccionario )
  # TODO: implementar

    def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
        columns = np.column_stack(datostrain)
        probabilidades = []
        clas = columns[-1]

        for col in columns[:-1]:

          if atributosDiscretos[columns.index(col)] == "Nominal":

            prob_clas = {}
            for i in range(len(col)):

                if col[i] not in prob_clas.keys():

                    prob_clas.update(col[i]:{"T":0,"F":0})

                prob_clas[col[i][clas[i]]] += 1

            for i in col:
                sum = prob_clas[col[i]["T"]] + prob_clas[col[i]["F"]]
                prob_clas[i["T"]] = prob_clas[i["T"]] / sum
                prob_clas[i["F"]] = prob_clas[i["F"]] / sum
            probabilidades.append(prob_clas)


          else:
            pass

        self.probabilidades = probabilidades
	    pass



  # TODO: implementar
  def clasifica(self,datostest,atributosDiscretos,diccionario):
        clasificador = []
        for fila in atributosDiscretos:
            prob = {"T":1,"F":1}
            for i in range(len(fila)):
                for key in prob:
                    prob[key] = prob[key] * self.probabilidades[i[key]]
            clasificador.append(prob)
        self.clasificacion = clasificador
    pass
