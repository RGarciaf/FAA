
# coding: utf-8

# In[ ]:


from Datos import Datos
from EstrategiaParticionado import *
from Clasificador import *
import numpy as np
from tabulate import tabulate
import pprint
from Roc import *

def medias(roc):
    TPR = {}
    FPR = {}
    medias = []
    ini = {}
    for clas in roc[0][0]:
        ini.update({clas:0})

    for archivo in roc:
        TPR = ini.copy()
        FPR = ini.copy()
        for particion in archivo:
            for clas in particion:
                TPR[clas] += particion[clas]["TPR"]
                FPR[clas] += particion[clas]["FPR"]
        for clas in TPR:
            TPR.update({clas:round(TPR[clas]/len(roc[0]),3)})
            FPR.update({clas:round(FPR[clas]/len(roc[0]),3)})
        medias.append({"TPR":TPR,"FPR":FPR})
    pprint.pprint(medias)
    return medias

# In[ ]:


# dataset = Datos("ConjuntosDatos/german.data")
# estrategiaS = ValidacionSimple()
# print("Estrategia Validacion Simple:")
# clasi = ClasificadorNaiveBayes()
# print(clas.roc(estrategiaS,dataset,clas))
r = Roc()
simple, cruzada, bootstrap = r.medias_roc()

medias(simple)
medias(cruzada)
medias(bootstrap)



# estrategiaS.creaParticiones(dataset)
# for i in estrategiaS.particiones:
#     # print("test: ", estrategiaS.particiones[i].indicesTest)
#     # print("train: ", estrategiaS.particiones[i].indicesTrain)
#
#     clas.entrenamiento(dataset.extraeDatos(i.indicesTrain), dataset.tipoAtributos, dataset.diccionarios)
# # print(np.column_stack(dataset.extraeDatos(range(950)))[-1])
#     clas.clasifica(dataset.extraeDatos(i.indicesTest), dataset.tipoAtributos, dataset.diccionarios)
#     print("Error: ",clas.error(dataset.extraeDatos(i.indicesTest), clas.clasificacion), "\n")

    # clas.entrenamiento(dataset.extraeDatos(estrategiaS.particiones[i].indicesTrain), dataset.tipoAtributos, dataset.diccionarios)
    # print(clas.probabilidades)
    # print(np.column_stack(dataset.extraeDatos(estrategiaS.particiones[i].indicesTest))[-1])
    # clas.clasifica(dataset.extraeDatos(estrategiaS.particiones[i].indicesTest), dataset.tipoAtributos, dataset.diccionarios)
    # print("\n")
# clas = ClasificadorNaiveBayes()
# clas.entrenamiento(dataset.extraeDatos([0,1,2,3,4,5,6,7,8,9]), dataset.tipoAtributos, dataset.diccionarios)
# print (clas.probabilidades)
# estrategiaC = ValidacionCruzada('ValidacionCruzada', 5, dataset)
# print("\nEstrategia Validacion Cruzada:")
# for i in estrategiaC.particiones.keys():
#     print("test: ", estrategiaC.particiones[i].indicesTest)
#     print("train: ", estrategiaC.particiones[i].indicesTrain)
