from abc import ABCMeta,abstractmethod
import numpy as np
from scipy.stats import norm
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
        clases = np.column_stack(datos)[-1]
        error = 0
        for clase, clase_pred in zip(clases, pred):
            if clase != list(clase_pred.keys())[0]:
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
    #     # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    #     # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    #     # y obtenemos el error en la particion de test i
    #     # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    #     # y obtenemos el error en la particion test



##############################################################################

class ClasificadorNaiveBayes(Clasificador):

    def __init__(self, laplace = False):
        self.probabilidades = []
        self.clasificacion = []
        self.laplace = laplace
        self.prior = {}

  # TODO: implementar

    def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
        columns = np.column_stack(datostrain)
        probabilidades = []
        clas = columns[-1]
        c, counts = np.unique(columns[-1], return_counts=True)
        p_class = dict(zip(c.astype(int), counts))

        if (self.laplace):
            ini_clas = dict.fromkeys(np.unique(columns[-1]).astype(int), 1)
            for key in p_class:
                p_class[key] += 1
            self.prior = dict(zip(c.astype(int), ((counts + 1) / (len(datostrain) + len(p_class)))))
        else:
            ini_clas = dict.fromkeys(np.unique(columns[-1]).astype(int), 0)
            self.prior = dict(zip(c.astype(int), counts / len(columns)))


        for key in diccionario:
            if key == "Class":
                pass
            else:
                prob_clas = {}
                if atributosDiscretos[list(diccionario.keys()).index(key)] == "Nominal":
                    for value in diccionario[key]:
                        prob_clas.update({int(diccionario[key][value]):ini_clas.copy()})
                else:
                    for value in diccionario["Class"]:
                        prob_clas.update({int(diccionario["Class"][value]):[]})
                probabilidades.append(prob_clas)


        for index_col in range(len(columns[:-1])):

            for i in range(len(columns[index_col])):
                if atributosDiscretos[index_col] == "Nominal":
                    probabilidades[index_col][columns[index_col][i]][int(clas[i])] += 1
                else:
                    probabilidades[index_col][int(clas[i])].append(columns[index_col][i])

            if atributosDiscretos[index_col] == "Nominal":
                for value in probabilidades[index_col]:
                    for i in probabilidades[index_col][value]:
                        probabilidades[index_col][value][i] = probabilidades[index_col][value][i] / p_class[i]
            else:
                for key in probabilidades[index_col]:
                    probabilidades[index_col][key] = {"media":np.mean(probabilidades[index_col][key]),"dp":np.std(probabilidades[index_col][key])}

        self.probabilidades = probabilidades

    def clasifica(self,datostest,atributosDiscretos,diccionario):

        prob_ini = {}
        for clas in diccionario["Class"]:
            prob_ini.update({diccionario["Class"][clas]:1})

        clasificacion = []
        printer = []
        for filad in datostest:
            prob = prob_ini.copy()
            for value, pattr in zip(filad,self.probabilidades):
                for clas in prob_ini:
                    if atributosDiscretos[self.probabilidades.index(pattr)] == "Nominal":

                        prob[clas] *= pattr[value][clas] * self.prior[clas]
                    else:
                        prob[clas] *= norm.pdf(value, pattr[clas]["media"], pattr[clas]["dp"])

            suma = np.sum(list(prob.values()))
            max = 0
            decision = ""
            for clas in diccionario["Class"]:
                if suma == 0:
                    prob[diccionario["Class"][clas]] = 0
                else:
                    prob[diccionario["Class"][clas]] = prob[diccionario["Class"][clas]]/suma
                if max <= prob[diccionario["Class"][clas]]:
                    max = prob[diccionario["Class"][clas]]
                    decision = clas
            clasificacion.append({diccionario["Class"][decision]:round(max,3)})
            printer.append({decision:round(max,3)})

        self.clasificacion = clasificacion
        return clasificacion

    #
    # def clasifica(self, datostest,atributosDiscretos,diccionario):
    #     columns = np.column_stack(datostest)
    #     c, counts = np.unique(columns[-1], return_counts=True)
    #     p_class = dict(zip(c.astype(int), counts))
    #     ini_clas = dict.fromkeys(np.unique(columns[-1]).astype(int), 0)
    #     clasifica = {}
    #     cla = []
    #     clas = columns[-1]
    #     for index_col in range(len(columns[:-1])):
    #         if atributosDiscretos[index_col] == "Nominal":
    #             for value in self.probabilidades[index_col]:
    #                 posterior, num, clasifica = [], [] , {}
    #                 for i in self.probabilidades[index_col][value]:
    #
    #                     verosimilitud = self.probabilidades[index_col][value][i]
    #                     prior = p_class[i] / len(columns[index_col])
    #                     num.append(verosimilitud * prior)
    #                 den = sum(num)
    #                 if (den > 0):
    #                     for x in num:
    #                         posterior.append(round (x / den, 2))
    #                     clasifica.update({posterior.index(max(posterior)):max(posterior)})
    #                 cla.append(clasifica)
    #
    #         else:
    #             norm.pdf(x,mean,std)
    #     return cla
    #
    #     #print(clasificador)