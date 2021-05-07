#-----------------------------------------------------------------------------------------------------
# 
#-----------------------------------------------------------------------------------------------------
# Predecir si una persona esta en capacidad de comprar casa o no, con el modelo K-NN
#-----------------------------------------------------------------------------------------------------
# Eliminar variables innecesarias
#-----------------------------------------------------------------------------------------------------
rm(list = ls())


#-----------------------------------------------------------------------------------------------------
# Explicaci?n del set de datos.
#-----------------------------------------------------------------------------------------------------
# Ingresos      : ingresos de la familia mensual.
# Gastos comunes: pagos de luz, agua, gas, etc mensual
# PagoVehiculo  : si se est? pagando cuota por uno o m?s veh?culos, y gastos en combustible al mes.
# gastos_otros  : compra en supermercado y lo necesario para vivir al mes
# Ahorros       : suma de ahorros dispuestos a usar para la compra de la casa.
# Vivienda      : precio de la vivienda que quiere comprar esa familia
# EstadoCivil   : 0-soltero  1-casados  2-divorciados
# Hijos         : cantidad de hijos menores y que no trabajan.
# Trabajo       : 0-sin empleo
#                 1-aut?nomo (freelance)
#                 2-empleado
#                 3-empresario
#                 4-pareja: aut?nomos
#                 5-pareja: empleados
#                 6-pareja: aut?nomo y asalariado
#                 7-pareja:empresario y aut?nomo
#                 8-pareja: empresarios los dos o empresario y empleado
# Comprar       : 0-No comprar
#                 1-Comprar (esta ser? nuestra columna de salida, para aprender)
#-----------------------------------------------------------------------------------------------------
# Cargar los datos.
#-----------------------------------------------------------------------------------------------------

Data <- read.csv("E:/Curso Data Science/Machine Learning - Modelos de Clasificacion/Machine_Learning_Modelos_de_Clasificacion/CompraAlquiler.csv",
                 sep = ",", 
                 dec = ".", 
                 stringsAsFactors = FALSE, 
                 na="NA")



#-----------------------------------------------------------------------------------------------------
# Explorar los datos.
#-----------------------------------------------------------------------------------------------------
str(Data)
summary(Data)

#-----------------------------------------------------------------------------------------------------
# Transformar los datos.
#-----------------------------------------------------------------------------------------------------

Analysis <- Data

Analysis$Comprar <- factor(Analysis$Comprar, 
                           levels = c(0,1),
                           labels = c('No', 'Yes'))


Normalizar <- function(x){
  return((x-min(x)) / (max(x) - min(x)))
}

Analysis <- cbind(Normalizar(Analysis[2:10]),
                  Analysis[11])

summary(Analysis)

#---------------------------------------------------------------------------------------------------
# Opcional: Exploraci?n de datos adicional (si quieres ver correlaciones, distribuciones, etc)
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
# Crear conjuntos de datos de entrenamiento y de prueba.
#---------------------------------------------------------------------------------------------------


library(caTools)
set.seed(100)
Div.Observaciones <- sample.split(Analysis$Comprar, SplitRatio = 0.7)
Datos.Entrenamiento <- Analysis[Div.Observaciones,]
Datos.Prueba <- Analysis[!Div.Observaciones,]
rm(Div.Observaciones)

#-----------------------------------------------------------------------------------------------------
# Crea probabilidades con los datos hist?ricos y asigna predicci?n a los nuevos clientes
#-----------------------------------------------------------------------------------------------------
# El modelo se contruye con los datos de entrenamiento y el valor m?ximo optimo de K. Es importante
# tener en cuenta que el modelo debe calibrarse para obtener el mejor resultado.
#-----------------------------------------------------------------------------------------------------

library(class)
clase_compra <- Datos.Entrenamiento$Comprar

compra_pred <- knn(train = Datos.Entrenamiento[-10], test = Datos.Prueba[-10],
                   cl= clase_compra)

# MAtriz de confusion 

compra_real <- Datos.Prueba$Comprar
table(compra_pred, compra_real)

#Calcular  la exactitud

mean(compra_pred == compra_real)

#------------------------------------------------------------------------------------
## COMPARACION CON DIFERENTE CANTIDAD DE VECINOS K ####
#------------------------------------------------------------------------------------

k1 <- knn(train = Datos.Entrenamiento[-10], test = Datos.Prueba[-10],
          cl= clase_compra)
mean(compra_real == k1)

k5 <- knn(train = Datos.Entrenamiento[-10], test = Datos.Prueba[-10],
          cl= clase_compra, k=5)
mean(compra_real == k5)

k9 <- knn(train = Datos.Entrenamiento[-10], test = Datos.Prueba[-10],
          cl= clase_compra, k=9)
mean(compra_real == k9)


# --------------------------------------------------------------------------------------
## MATRIZ DE CONFUSION
# --------------------------------------------------------------------------------------




