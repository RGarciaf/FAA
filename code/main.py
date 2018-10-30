
# coding: utf-8

# In[ ]:


from Datos import Datos
from EstrategiaParticionado import *
from Clasificador import *
import numpy as np
from tabulate import tabulate
import pprint


# In[ ]:


dataset = Datos("ConjuntosDatos/balloons.data")
estrategiaS = ValidacionSimple()
print("Estrategia Validacion Simple:")
clas = ClasificadorNaiveBayes()

pprint.pprint(clas.roc(estrategiaS,dataset,clas))

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
