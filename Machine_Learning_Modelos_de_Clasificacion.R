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

#--------------------------------------------------------------------
#COMPARACION CON DIFERENTE CANTIDAD DE VECINOs K
#------------------------------------------------------------------

k1 <- knn(train = Datos.Entrenamiento[-13], test = Datos.Prueba[-13], cl= tipo_vino)
mean(vino_real == k1)

k7 <- knn(train = Datos.Entrenamiento[-13], test = Datos.Prueba[-13], cl= tipo_vino, k=7)
mean(vino_real == k7)

k13 <- knn(train = Datos.Entrenamiento[-13], test = Datos.Prueba[-13], cl= tipo_vino, k=13)
mean(vino_real==k13)


#--------------------------------------------------------------------
# OBTENER LAS PROBABILIDADES
#----------------------------------------------------------------------

vino_pred_prob <- k1 <- knn(train = Datos.Entrenamiento[-13], test = Datos.Prueba[-13], cl= tipo_vino, k=7, prob = TRUE)

vino_prob <- attr(vino_pred_prob, "prob")

head(vino_pred_prob)

head(vino_prob)

summary(vino_prob)


#--------------------------------------------------------------------
# NORMALIZAR LAS VARIABLES
# ------------------------------------------------------------------

normalizar <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}

Datos.Train.Scale <- normalizar(Datos.Entrenamiento[1:12])
Datos.Train.Scale <- cbind(Datos.Train.Scale, Datos.Entrenamiento$type)

colnames(Datos.Train.Scale)[13] <- "type"

summary(Datos.Entrenamiento)
summary(Datos.Train.Scale)

table(Datos.Entrenamiento$type)
table(Datos.Train.Scale$type)

Datos.Test.Scale <- normalizar(Datos.Prueba[1:12])
Datos.Test.Scale <- cbind(Datos.Test.Scale, Datos.Prueba$type)

colnames(Datos.Test.Scale)[13] <- "type"

# Repetir el analisis KNN 

k7_scale <- knn(train = Datos.Train.Scale[-13], test = Datos.Test.Scale[-13], cl = tipo_vino, k=7)
mean(vino_real == k7_scale)

vino_prediccion_k7 <- knn(train = Datos.Entrenamiento[-13], test = Datos.Prueba[-13], 
                          cl = tipo_vino, k=7)

# Y crearemos una matriz de confusión

table(vino_prediccion_k7, vino_real)

vino_prediccion_k7_scale <- knn(train = Datos.Train.Scale[-13], test = Datos.Test.Scale[-13], 
                                cl= tipo_vino, k=7, prob = TRUE)

vino_prob_k7Scale <- attr(vino_prediccion_k7_scale, "prob")


head(vino_prediccion_k7_scale)

head(vino_prob_k7Scale)

summary(vino_prob_k7Scale)


# Comparacion contra datos normalizados

vino_prediccion_k7_prueba <- knn(train = Datos.Train.Scale[-13], test = Datos.Test.Scale[-13], 
                                 cl= tipo_vino, k=7)
table(vino_prediccion_k7_prueba, vino_real)
