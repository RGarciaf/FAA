
# coding: utf-8

# In[ ]:


from Datos import Datos
from EstrategiaParticionado import *
from Clasificador import *
import numpy as np
from tabulate import tabulate
import pprint
from Roc import *




# dataset = Datos("ConjuntosDatos/balloons.data")
# reglog = ClasificadorRegresionLogistica()
# reglog.entrenamiento(dataset.datos[:10],dataset.nominalAtributos,dataset.diccionarios)
# pprint.pprint(reglog.clasifica(dataset.datos[10:],dataset.nominalAtributos,dataset.diccionarios ))

print("\nBalloons:")
dataset = Datos("ConjuntosDatos/balloons.data")
reglog = ClasificadorRegresionLogistica()
print("\nRegresion Logistica")
pprint.pprint(reglog.validacion(ValidacionSimple(),dataset,reglog))

knn = ClasificadorVecinosProximos(5)
print("\nVecinos Proximos")
pprint.pprint(knn.validacion(ValidacionSimple(),dataset,knn))

clas = ClasificadorNaiveBayes()
print("\nNaive Bayes")
pprint.pprint(clas.validacion(ValidacionSimple(),dataset,clas))


print("\n\nGerman:")
dataset = Datos("ConjuntosDatos/german.data")
reglog = ClasificadorRegresionLogistica()
print("\nRegresion Logistica")
pprint.pprint(reglog.validacion(ValidacionSimple(),dataset,reglog))

knn = ClasificadorVecinosProximos(5)
print("\nVecinos Proximos")
pprint.pprint(knn.validacion(ValidacionSimple(),dataset,knn))

clas = ClasificadorNaiveBayes()
print("\nNaive Bayes")
pprint.pprint(clas.validacion(ValidacionSimple(),dataset,clas))

# dataset = Datos("ConjuntosDatos/german.data")
# knn = ClasificadorNaiveBayes()
# knn.entrenamiento(dataset.datos[:500],dataset.nominalAtributos,dataset.diccionarios)
# pprint.pprint(knn.clasifica(dataset.datos[:500],dataset.nominalAtributos,dataset.diccionarios ))



# dataset = Datos("ConjuntosDatos/german.data")
# estrategiaS = ValidacionSimple()
# print("Estrategia Validacion Simple:")
# clasi = ClasificadorNaiveBayes()
# print(clas.roc(estrategiaS,dataset,clas))
# r = Roc()
# r.medias_roc()
# r.calcula_medias_roc()
# pprint.pprint(r.simple)
# pprint.pprint(r.cruzada)
# pprint.pprint(r.bootstrap)

# for particion in r.simple[0]:
#     for


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
