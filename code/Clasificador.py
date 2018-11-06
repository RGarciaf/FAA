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

class ClasificadorVecinosProximos(Clasificador):

    def __init__(self, vecinos = 1):
        self.mediaDesvAtributos = []
        self.datosNormalizados = []
        self.vecinos = vecinos
        self.clasificacion = []
        self.datos = []

    def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
        self.calcularMediasDesv(datosTrain)
        self.datosNormalizados = self.normalizarDatos(datosTrain)
        self.datos = datosTrain
        
    def clasifica(self,datosTest,atributosDiscretos,diccionario):
        datosTest_normalizados = self.normalizarDatos(datosTest)

        sumas = []
        for fila_test in datosTest_normalizados:                       
            self.clasificacion.append(self.extraeClase(fila_test))
        
        return self.clasificacion
            

    def extraeProb(self,vecinos):
        clas = {}
        for vecino in vecinos:
            clas[vecino[-1]] = 0

        for vecino in vecinos:
            clas[vecino[-1]] += 1

        for value in clas:
            clas[value] = round(clas[value]/len(vecinos), 3)
        return clas


    def extraeClase(self, fila_test):
        datos_norm_numpy = self.datosNormalizados
        datos_norm_numpy = pow(datos_norm_numpy, 2) 

        fila_test_numpy = np.array(fila_test)
        fila_test_numpy = pow(fila_test_numpy, 2)

        resta =  np.absolute(datos_norm_numpy - fila_test_numpy) ** 0.5
        array = []
        for fila, i in zip(resta, range(len(resta))):
            array.append(fila.tolist())
            array[i].append(fila.sum())
            array[i].append(self.datos[i][-1])
            # np.append(resta[i],fila.sum())
            # np.append(resta[i],self.datos[i][-1])
            # print("array> ", array[i])
            # print("fila> ", fila, fila.sum)
            # print(resta[i])

        vecinos = sorted(array, key = lambda x: x[-2])[:self.vecinos]

        return self.extraeProb(vecinos)

        



    def calcularMediasDesv(self, datostrain):
        columns = np.column_stack(datostrain)[:-1]
        meanStdAttrs = []
        for i in range(len(columns)):
            meanStdAttrs.append([round(np.mean(columns[i]), 3), round(np.std(columns[i]), 3)])
        self.mediaDesvAtributos = meanStdAttrs
    
    def normalizarDatos(self, datos):
        columns = np.column_stack(datos)
        # datosNorm = np.zeros((len(columns), len(datos)))
        datosNorm = []
        for col, mdv in zip(columns, self.mediaDesvAtributos):
            fila = []
            for value_col in col:
                fila.append( round(value_col - mdv[0]/mdv[1], 3))
            datosNorm.append(fila)

        # for i in range(len(columns)):
        #     for j in range(len(datos)):
        #         datosNorm[i][j] = round((columns[i][j] - self.mediaDesvAtributos[i][0]) / self.mediaDesvAtributos[i][1], 3)

        return  np.column_stack(np.array(datosNorm))




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
            # print("\n\nIteracion ")
            prob = prob_ini.copy()
            for value, pattr in zip(filad,self.probabilidades):
                for clas in prob_ini:
                    if atributosDiscretos[self.probabilidades.index(pattr)] == "Nominal":
                        # print("Nominal>",value,clas,pattr[value])
                        prob[clas] *= pattr[value][clas] * self.prior[clas]
                    else:
                        # print("Continuo>",value,clas, pattr[clas])
                        prob[clas] *= norm.pdf(value, pattr[clas]["media"], pattr[clas]["dp"])

            suma = np.sum(list(prob.values()))
            max = -1
            decision = ""
            for clas in diccionario["Class"]:
                if suma == 0:
                    prob[diccionario["Class"][clas]] = 0
                else:
                    prob[diccionario["Class"][clas]] = prob[diccionario["Class"][clas]]/suma
                # print(clas, decision, max, prob[diccionario["Class"][clas]])
                if max <= prob[diccionario["Class"][clas]]:
                    max = prob[diccionario["Class"][clas]]
                    decision = clas
                    # print("if>",clas, decision)
            # print("decision, max>",decision, max)
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
