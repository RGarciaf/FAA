import numpy as np
import math

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
        self.nominalAtributos = self.attrToIndex()
        self.mediaDesvAtributos = {}
        self.k = None
        self.a = None
        self.datosIntervalizados = []
    
        

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
    
    # Diccionario, para cada clave (posicion de atributo), [media, std]
    def calcularMediasDesv(self, datostrain):
        columns = np.column_stack(datostrain)
        meanStdAttrs = {}
        for i in range(len(columns)):
            meanStdAttrs[i] = [round(np.mean(columns[i]), 3), round(np.std(columns[i]), 3)]        
        self.mediaDesvAtributos = meanStdAttrs
    
    def normalizarDatos(self, datos):
        columns = np.column_stack(datos)
        datosNorm = np.zeros((len(columns), len(datos)))
        for i in range(len(columns)):
            for j in range(len(datos)):
                datosNorm[i][j] = round((columns[i][j] - self.mediaDesvAtributos[i][0]) / self.mediaDesvAtributos[i][1], 3)

        return np.column_stack(datosNorm)
        
    def crearIntervalos(self, datos):
        #K = 1 + 3.322 log10 N 
        n = np.column_stack(datos)
        log = math.floor(math.log(len(datos),10))
        k = int(1 + 3.322 * log)
        #A = (Xmax – Xmin) / K 
        a = []
        for row in n[:-1]:
            a.append((np.amax(row) - np.amin(row))/k)
        self.k = k
        self.a = a
        return k, a
    
    
    def convertirAIntervalos(self, datos):
        columns = np.column_stack(datos)
        datosIntervalos = np.zeros((len(columns)-1, len(datos))) #Dos parentesis, tiene que ser un shape
        mins = []
        for row in columns[:-1]: #Necesito los minimos de cada atributo
            mins.append(np.amin(row))
            
        for i, (fila_inter, col, a, col_min) in enumerate(zip(datosIntervalos, columns[:-1], self.a, mins)): #Itero sobre los datos correspondientes a las columnas
            for j,(attr_inter, dato) in enumerate(zip(fila_inter, col)): #por cada valor de la fila, uso el minimo de ese atributo y el "a" de ese atributo
                fila_inter[j] = math.ceil((dato - col_min)/a)
                
        self.datosIntervalizados = np.column_stack(datosIntervalos.tolist() + [columns[-1]])
        return np.column_stack(datosIntervalos.tolist() + [columns[-1]])