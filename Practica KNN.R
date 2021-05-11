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

Analysis$EstadoCivil <- factor(Analysis$EstadoCivil, 
                               levels = c(0,1,2),
                               labels = c('Soltero', 'Casado', 'Divorciado'))


Analysis$Trabajo <- factor(Analysis$Trabajo, 
                           levels = c(0,1,2,3,4,5,6,7,8),
                           labels = c('Sin Empleo', 
                                      'Freelance',
                                      'Empleado',
                                      'Empresario',
                                      'Nucleo Freelance',
                                      'Nucleo Empleados',
                                      'Freelance y Asalariado',
                                      'Empresario y Freelance',
                                      'Mixto'))



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
install.packages("kknn")
library(kknn)

datos.Modelo.KNN <- train.kknn(Comprar ~ ., data = Datos.Entrenamiento[-1], kmax = 10)
#------------------------------------------------------------------------------------
## Se crean las predicciones
#------------------------------------------------------------------------------------

Datos.Modelo.KNN.Prediccion <- predict(datos.Modelo.KNN, Datos.Prueba[, -c(1,11)])
Datos.Modelo.KNN.Prediccion

MatrizConfusion <- table(Datos.Prueba[,11], Datos.Modelo.KNN.Prediccion, dnn = c("Real", "Prediccion"))
MatrizConfusion

round(prop.table(MatrizConfusion)*100, 2)
round(prop.table(MatrizConfusion, 1)*100, 2)
round(prop.table(MatrizConfusion, 2)*100, 2)
# --------------------------------------------------------------------------------------
## MATRIZ DE CONFUSION
# --------------------------------------------------------------------------------------

library(ggplot2)

library(reshape2)

x <- melt(MatrizConfusion)
x

ggplot(x, aes(Real, Prediccion))+
  geom_point(aes(size=value), alpha=0.8, color="darkblue", show.legend = FALSE)+
  geom_text(aes(label = value), color="white")+
  scale_size(range=c(15,50))+
  theme_bw()
