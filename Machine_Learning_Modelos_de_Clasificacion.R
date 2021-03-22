## Modelo K-Nearest Neighbors KNN 

# Aprenderemos a clasificar el tipo de vino (Tinto o Blanco)
#De acuerdo a sus caracteristicas quimicas

# Carga de datos de vino

vinos_original <- read.csv(file = "E:/Curso Data Science/Machine Learning - Modelos de Clasificacion/wine red-white.csv", 
                           header = TRUE, sep = ";", dec = ".", stringsAsFactors = FALSE, na = "NA")


vinos <- vinos_original

#Analizar las primeras lineas
head(vinos)

#Revisar la estructura de datos
str(vinos)
str(vinos)


install.packages("class")
library(class)

# Cuantas observaciones tenemos de cada tipo
table(vinos$type)

# En promedio, cual es el valor de cada tipo de vino

aggregate(pH ~ type, data = vinos, mean)
aggregate(alcohol ~ type, data = vinos, mean)
aggregate(total.sulfur.dioxide ~ type, data = vinos, mean)

## Construccion del modelo KNN

# Vamos a crear los conjuntos de datos de entrenamiento y prueba

install.packages("caTools")
library(caTools)


set.seed(1000)

Div.Observaciones <- sample.split(vinos$type, SplitRatio = 0.7)

Datos.Entrenamiento <- vinos[Div.Observaciones,]

Datos.Prueba <- vinos[!Div.Observaciones,]

vinos_test <- vinos

rm(Div.Observaciones)

# Ahora utilizaremos el modelo KNN para crear las predicciones

# Crear un vector con las etiquetas (En este caso vino)
install.packages("class")
library(class)
tipo_vino <- Datos.Entrenamiento$type

vino_prediccion <- knn(train = Datos.Entrenamiento[-13], test = Datos.Prueba[-13], cl= tipo_vino)

# Y Creamos una Matriz de confusion

vino_real <- Datos.Prueba$type
table(vino_prediccion, vino_real)

#Calcular la exactitud
mean(vino_prediccion == vino_real)

