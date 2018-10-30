import numpy as np

class Datos ( object ):
    TiposDeAtributos = ('Continuo','Nominal')

    def attrsToIndex():
        nominal = []
        for atributo in Datos.tipoAtributos:
            if(atributo == "Nominal"):
                nominal.append(1)
            else:
                nominal.append(0)
        self.nominalAtributos = nominal
        
    def parseMetaDatos(fichero):
        """  Parsea las tres primeras lineas del fichero  """

        numAtrs = int(fichero.readline())
        nombreAtributos = fichero.readline().replace("\n","").split(",")
        tipoAtributos = fichero.readline().replace("\n","").split(",")
        return numAtrs, nombreAtributos, tipoAtributos

    def iniDicc(nombreAtributos):
        """  Crea un diccionario con los nombres de los atributos como claves  """

        diccionarios = {}
        for atr in nombreAtributos:
            diccionarios[atr] = []
        return diccionarios

    def insertAtr(fichero, nombreAtributos):
        """  Rellena el diccionario con los posibles valores de cada atributo  """

        diccionarios = Datos.iniDicc(nombreAtributos)
        for linea in fichero:
            valor = linea.replace("\n","").split(",")

            for i in range(len(valor)):
                if valor[i] not in diccionarios[nombreAtributos[i]]:
                    diccionarios[nombreAtributos[i]].append(valor[i])
        return diccionarios

    def transformaADicc(fichero, nombreAtributos, tipoAtributos):
        """  Rellena el diccionario con una codificacion de numeros
            para cada valor de los atributos nominales
        """

        diccionarios = Datos.insertAtr(fichero, nombreAtributos)
        for atr in diccionarios:
            valores = sorted(diccionarios[atr])

            diccionarios[atr] = {}
            if tipoAtributos[nombreAtributos.index(atr)] == 'Nominal':
                for valor in valores:
                    diccionarios[atr].update({valor:valores.index(valor)})
        return diccionarios

    def extraerMetaDatos(nombreFichero):
        """  Genera:
            un diccionario de la codificacion de los valores,
            un array con los nombres de los atributos y
            un array con el tipo de los atributos
        """

        fichero = open(nombreFichero, "r")
        numAtrs, nombreAtributos, tipoAtributos = Datos.parseMetaDatos(fichero)
        diccionarios = Datos.transformaADicc(fichero, nombreAtributos, tipoAtributos)
        fichero.close()
        return diccionarios, numAtrs, nombreAtributos, tipoAtributos

    def transformDataToArray(nombreFichero):
        """  Genera un array de datos codificando los valores acorde al diccionario creado
        """

        diccionarios, numAtrs, nombreAtributos, tipoAtributos = Datos.extraerMetaDatos(nombreFichero)
        fichero = open(nombreFichero, "r")

        lineas = fichero.readlines()[3:]
        datos = np.empty( (numAtrs,len(nombreAtributos)))

        for j in range(numAtrs):
            valores = lineas[j].replace("\n","").split(",")

            for i in range(len(valores)):
                if tipoAtributos[i] == 'Nominal':
                    datos[j][i] = Datos.buscarDiccionario(diccionarios,valores[i])
                else:
                    datos[j][i] = valores[i]

        fichero.close()
        return datos, diccionarios, nombreAtributos, tipoAtributos

    def buscarDiccionario(dic, valor):
        """  Busca la codificacion de un valor en el diccionario
        """

        for atr in dic:
            if valor in dic[atr]:
                return dic[atr][valor]


    def __init__ ( self, filename ):

        self.datos, self.diccionarios, self.nombreAtributos, self.tipoAtributos = Datos.transformDataToArray(filename)
        self.nominal = self.attrToIndex()
       
        pass

    
        

    def extraeDatos(self, idx):
        subconjunto = []
        for i in idx:
            subconjunto = np.append(subconjunto, self.datos[i])
        subconjunto = subconjunto.reshape((len(idx), len(self.nombreAtributos)))
        return subconjunto
        
    def attrToIndex(self):
        nominal = []
        for atributo in self.tipoAtributos:
            if(atributo == "Nominal"):
                nominal.append(1)
            else:
                nominal.append(0)
        return nominal