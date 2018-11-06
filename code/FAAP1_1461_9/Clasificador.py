from abc import ABCMeta,abstractmethod
import numpy as np
from scipy.stats import norm
import pprint
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

    @abstractmethod
    def roc(self,particionado,dataset,clasificador):
        pass



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
            # #print("\n\nIteracion ")
            prob = prob_ini.copy()
            for value, pattr in zip(filad,self.probabilidades):
                for clas in prob_ini:
                    if atributosDiscretos[self.probabilidades.index(pattr)] == "Nominal":
                        # #print("Nominal>",value,clas,pattr[value])
                        prob[clas] *= pattr[value][clas] * self.prior[clas]
                    else:
                        # #print("Continuo>",value,clas, pattr[clas])
                        prob[clas] *= norm.pdf(value, pattr[clas]["media"], pattr[clas]["dp"])

            suma = np.sum(list(prob.values()))
            max = -1
            decision = ""
            for clas in diccionario["Class"]:
                if suma == 0:
                    prob[diccionario["Class"][clas]] = 0
                else:
                    prob[diccionario["Class"][clas]] = prob[diccionario["Class"][clas]]/suma
                # #print(clas, decision, max, prob[diccionario["Class"][clas]])
                if max <= prob[diccionario["Class"][clas]]:
                    max = prob[diccionario["Class"][clas]]
                    decision = clas
                    # #print("if>",clas, decision)
            # #print("decision, max>",decision, max)
            clasificacion.append({diccionario["Class"][decision]:round(max,3)})
            printer.append({decision:round(max,3)})

        self.clasificacion = clasificacion
        return clasificacion

    def roc(self,particionado,dataset,clasificador):

        particionado.creaParticiones(dataset)
        clase_ini = {}
        for clase in dataset.diccionarios["Class"].values():
            clase_ini.update({clase:0.0000000001})

        matriz = []
        for particion in particionado.particiones:
            clasificador.entrenamiento(dataset.extraeDatos(particion.indicesTrain), dataset.tipoAtributos, dataset.diccionarios)
            clasificacion = clasificador.clasifica(dataset.extraeDatos(particion.indicesTest), dataset.tipoAtributos, dataset.diccionarios)

            clases = np.column_stack(dataset.datos)[-1].astype(int)

            falsos = clase_ini.copy()
            verdaderos = clase_ini.copy()
            roc = clase_ini.copy()

            for clase, clase_pred in zip(clases, clasificacion):
                if clase == list(clase_pred.keys())[0]:
                    verdaderos[clase] += 1
                else:
                    falsos[clase] += 1
            for clase1 in dataset.diccionarios["Class"].values():
                fn = 0
                fp = falsos[clase1]
                tp = verdaderos[clase1]
                tn = 0
                for clase2 in dataset.diccionarios["Class"].values():
                    if clase1 != clase2:
                        fn += falsos[clase2]
                        tn += verdaderos[clase2]
                roc[clase1]={"TPR":round(tp / (fn + tp),3),
                    "FNR":round(fn / (tp + fn),3),
                    "FPR":round(fp / (fp + tn),3),
                    "TNR":round(tn / (fp + tn),3),
                    "fn":round(fn,3),
                    "fp":round(fp,3),
                    "tp":round(tp,3),
                    "tn":round(tn,3) }
            matriz.append(roc)
        return matriz
