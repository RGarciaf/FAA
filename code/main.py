
# coding: utf-8

# In[ ]:


from Datos import Datos
from EstrategiaParticionado import *
from Clasificador import *
import numpy as np
from tabulate import tabulate


# In[ ]:


dataset = Datos("ConjuntosDatos/tic-tac-toe.data")
estrategiaS = ValidacionSimple('ValidacionSimple', 5, 0.5, dataset)
print(dataset.diccionarios)
print("Estrategia Validacion Simple:")
for i in estrategiaS.particiones.keys():
    # print("test: ", estrategiaS.particiones[i].indicesTest)
    # print("train: ", estrategiaS.particiones[i].indicesTrain)
    clas = ClasificadorNaiveBayes()

    clas.entrenamiento(dataset.extraeDatos(range(950)), dataset.tipoAtributos, dataset.diccionarios)
    print(clas.probabilidades)
    print(np.column_stack(dataset.extraeDatos(range(950)))[-1])
    clas.clasifica(dataset.extraeDatos(range(950)), dataset.tipoAtributos, dataset.diccionarios)
    print("\n")

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

