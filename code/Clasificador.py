from abc import ABCMeta,abstractmethod
import numpy as np
#from tabulate import tabulate

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
    # TODO: implementar
    def error(self,datos,pred):
    #     # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
        pass



    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    # TODO: implementar esta funcion
    def validacion(self,particionado,dataset,clasificador,seed=None):

    #     # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    #     # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    #     # y obtenemos el error en la particion de test i
    #     # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    #     # y obtenemos el error en la particion test
        pass


##############################################################################

class ClasificadorNaiveBayes(Clasificador):

    def __init__(self):
        self.probabilidades = []

  # TODO: implementar

    def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
        columns = np.column_stack(datostrain)
        probabilidades = []
        clas = columns[-1]
        ini_clas = {}
        c, counts = np.unique(columns[-1], return_counts=True)
        p_class = dict(zip(c.astype(int), counts))
       
        for key in diccionario["Class"]:
            ini_clas.update({int(diccionario["Class"][key]):0})
        for key in diccionario:
            if key == "Class":
                pass
            else:
                prob_clas = {}
                for value in diccionario[key]:
                    prob_clas.update({int(diccionario[key][value]):ini_clas.copy()})
                probabilidades.append(prob_clas)
        
        for index_col in range(len(columns[:-1])):

            if atributosDiscretos[index_col] == "Nominal":
                for i in range(len(columns[index_col])):      
                    probabilidades[index_col][columns[index_col][i]][int(clas[i])] += 1  
           
                
                for value in probabilidades[index_col]:
                   
                    for i in probabilidades[index_col][value]:
                        probabilidades[index_col][value][i] = round(probabilidades[index_col][value][i] / p_class[i], 2)


            else:
                pass

        self.probabilidades = probabilidades
        # attrs = list(diccionario.keys())
        # print (diccionario, probabilidades)
        # for attr in diccionario:
        #     values = list(diccionario[attr].keys())
        #     print (values)
        #     for value in diccionario[attr]:
        #         print (diccionario[attr][value])
        #         probabilidades[attrs.index(attr)][value] = probabilidades[attrs.index(attr)][diccionario[attr][value]].pop(diccionario[attr][value])


  # TODO: implementar
    def clasifica(self,datostest,atributosDiscretos,diccionario):
        clasificador = []
        # print("datos: ",datostest)
        # for filad, filap in zip(datostest, self.probabilidades):
        for filad in datostest:
            prob = {}
            for clas in diccionario["Class"]:
                prob.update({diccionario["Class"][clas]:1})

            for value, pattr in zip(filad,self.probabilidades):
                for clas in pattr[value]:
                    prob[clas] *= pattr[value][clas]
            # for valued in filad :
            #     for clas in filap[valued]:
            #         prob[clas] *= filap[valued][clas]
            suma = np.sum(list(prob.values()))
            max = 0
            decision = ""
            for clas in diccionario["Class"]:
                prob[diccionario["Class"][clas]] = prob[diccionario["Class"][clas]]/suma
                if max < prob[diccionario["Class"][clas]]:
                    max = prob[diccionario["Class"][clas]]
                    decision = clas
            clasificador.append({decision:max})
            
            
    def clasificaP(self, datostest,atributosDiscretos,diccionario):
        columns = np.column_stack(datostest)
        c, counts = np.unique(columns[-1], return_counts=True)
        p_class = dict(zip(c.astype(int), counts))

        clasifica = {}
        clas = columns[-1]
        for index_col in range(len(columns[:-1])):
            if atributosDiscretos[index_col] == "Nominal":
                for value in self.probabilidades[index_col]:   
                    posterior, num = [], []
                    for i in self.probabilidades[index_col][value]:
                       
                        verosimilitud = self.probabilidades[index_col][value][i] 
                        prior = p_class[i] / len(columns[index_col])
                        num.append(verosimilitud * prior)
                    den = sum(num)
                    if (den > 0):
                        for x in num:
                            posterior.append(round (x / den, 2))
                        print(posterior.index(max(posterior)), max(posterior),  posterior)
                        clasifica.update({posterior.index(max(posterior)):max(posterior)})
        
        return clasifica
                
        #print(clasificador)
