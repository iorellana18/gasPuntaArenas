### Predicción de consumo de gas en Punta Arenas

  ######################## Exploración de datos ######################## 

# Primero se importa el conjunto de datos y se imprimen los primeros ejemplares para observar su naturaleza.
gasConsume <- read.csv("gasConsume.csv", sep=";")
head(gasConsume)

# Se puede observar que los datos corresponden a dias, meses y años, un indicador de feriados y por cada fecha
# la magnitud del consumo de gas en valores contiuos.
# Luego se realizan visualizaciones para estudiar de forma preliminar el comportamiento de los datos.
# Se crea un vector con el formato de las fechas para realizar una correcta visualización
Fecha <- as.Date(paste(gasConsume$Mes,gasConsume$Dia,gasConsume$A.o, sep="/"),"%m/%d/%Y")

# Para visualizar los datos se utiliza la biblioteca ggplot que posee una interfaz sencilla para
# crear gráficos.
library(ggplot2)
ggplot(data=gasConsume, aes(x=Fecha, y=Consumo, group=1))+geom_point(color="blue")
# Se puede apreciar en este gráfico que existe cierto periodo a través de los años en cuanto al consumo de gas.

ggplot(data=gasConsume, aes(x=Mes, y=Consumo, group=1))+geom_point(color="blue")
# Los rangos de valores de gas consumido son claramente superiores en algunos meses, aún así van desde las 200
# a 400 unidades aproximadamente por lo que sería imposible predecir de esta forma el consumo para un día en 
# específico

ggplot(data=gasConsume, aes(x=Dia, y=Consumo, group=1))+geom_point(color="blue")
# No se aprecia ninguna tendencia en el consumo dependiendo el día

# Para modelar la forma en que el consumo varía dependiendo basándose en estas variables se debe crear un modelo 
# de regresión. Sin embargo los datos no se encuentran en la forma apropiada para ser modelados de la mejor manera.
# Por esto se realizara una fase de pre-procesamiento para transformar los datos y obtener los conjuntos de 
# entrenamiento y prueba.


######################## Pre-procesamiento ######################## 


# Antes de entrenar el modelo es recomendable utilizar el proceso one-hot encoding. Como el conjunto de datos
# posee valores categórico (los días de semana), lo adecuado es pasarlo a valores numéricos para facilitar el
# aprendizaje del algoritmo. Una solución podría ser renombrar esta variable con valores del 1 al 7 para 
# representar los días, pero esto podría conducir a que el algoritmo le de mayor peso al Domingo por ser el 
# número 7 que al Lunes por ser el 1, por lo tanto es recomendable convertir esta variable en una matriz de datos
# binarios. Para esto se utiliza la función dummyVars de la biblioteca caret que convierte cada valor de una 
#columa en una columna que indica la presencia o ausencia del valor.
library(caret)
dmy <- dummyVars(" ~ .", data = gasConsume)
gasConsumeNumeric <- data.frame(predict(dmy, newdata = gasConsume))
head(gasConsumeNumeric)
# Se puede apreciar como se crea una columna por cada valor categórico encontrado

# Luego es recomendable escalar los datos ya que poseen diversas escalas de magnitudes en las columnas lo que 
# provocará que algunos tengan mayor influencia sobre el modelo. Para esto se crea una pequeña función que
# dejará a las características en rangos de 0 a 1 (no es necesario hacer eso con la variable que se intenta 
# predecir)
range01 <- function(x){(x-min(x))/(max(x)-min(x))}
gasConsumeNumeric$Dia <- range01(gasConsumeNumeric$Dia)
gasConsumeNumeric$Mes <- range01(gasConsumeNumeric$Mes)
gasConsumeNumeric$A.o <- range01(gasConsumeNumeric$A.o)
head(gasConsumeNumeric)

# Luego hay que dividir el dataframe en los conjuntos de entrenamiento y prueba. Como se trata de fechas que se
# repiten de forma periodica hay que asegurarse de tener todo el espectro posible de valores tanto en el 
# conjunto de entrenamiento como en el de prueba (no debe haber un conjunto que no posea muestras de un mes
# por ejemplo). Para esto se utiliza la función createDataPartition de caret repartiendo los indices para equilibrar
# la cantidad de muestras de meses. Aún así es necesario probar que los datos porvienen de distribuciones similares
# más adelante. Para asegurar que se repetirá la misma prueba se utiliza la función set.seed() con el número 42
set.seed(42)
splitIndex <- createDataPartition(y=gasConsumeNumeric$Mes,p=0.75,list=FALSE)
trainSet <- gasConsumeNumeric[splitIndex,]
testSet <- gasConsumeNumeric[-splitIndex,]
fechaTrain <- Fecha[splitIndex]
fechaTest <- Fecha[-splitIndex]

# Grupo de entrenamiento
ggplot(data=trainSet, aes(x=fechaTrain, y=Consumo, group=1))+geom_point(color="blue")+ 
  ggtitle("Entrenamiento")+xlab("Fecha") 
# Grupo de prueba
ggplot(data=testSet, aes(x=fechaTest, y=Consumo, group=1))+geom_point(color="red")
  + ggtitle("Prueba")+xlab("Fecha") 
# Se puede apreciar que la muestra de test posee datos de todo el espectro de meses y años.

plot(density(trainSet$Consumo),main="Densidad por unidades de consumo",xlab="Unidades de consumo"
     ,col = "blue")
lines(density(testSet$Consumo), col="red")
legend("topright", legend=c("Entrenamiento", "Prueba"),
       col=c("blue", "red"), lty=1:2, cex=0.8)
# Las densidades de las unidades de consumo por grupo parecen ser bastante similares de igual forma.
# Para descartar una distribución muy alejada entre los datos se puede recurrir a test estadísticos.
# En caso de ser datos normales se puede recurrir al test de t-student, de otra forma al de Wilcox-Mann Whitney
# Para probar la normalidad se usa el test de Shapiro Wilk. Se aplicarán estos test en los valores de consumo 
# y mes cuya varianza puede resultar más riesgosa para el modelo.

print("Test de normalidad de consumo (entrenamiento)")
shapiro.test(trainSet$Consumo)
print("Test de normalidad de consumo (prueba)")
shapiro.test(testSet$Consumo)
print("Test de normalidad de consumo (entrenamiento)")
shapiro.test(trainSet$Mes)
print("Test de normalidad de consumo (prueba)")
shapiro.test(testSet$Mes)
# Todos los conjuntos dan como resultado un p-value menor a 0.05 que es la confianza habitual, por lo que claramente
# no pertenecen a una distribución normal

print("Test de Wilcoxon (consumo)")
wilcox.test(trainSet$Consumo,testSet$Consumo)
print("Test de Wilcoxon (mes)")
wilcox.test(trainSet$Mes,testSet$Mes)
# Como los p-values son superiores a 0.05 en ambos casos no se puede descartar la hipótesis de que los datos
# provengan de una distribución similar.

# Finalmente se reordenan los valores de forma aleatoria antes de entrenar el modelo
set.seed(42)
trainSet <- trainSet[sample(1:nrow(trainSet)), ]
set.seed(42)
testSet <- testSet[sample(1:nrow(testSet)), ]



######################## Modelamiento de datos ######################## 
### SVR ###
# El primer algoritmo de machine learning a utilizar es Support Vector Regression (SVR)
# Para esto se utiliza la biblioteca e1071
library(e1071)
# Para evaluar de forma preliminar los resultados se define una función de error RMSE
rmse <- function(error){
  sqrt(mean(error^2))
}

# Como SVR utiliza varios hyper-parámetros para ajsutar el modelo, se utiliza el método de Grid Search para buscar
# los que produzcan los mejores resultados posibles que consiste básicamente en probar una tupla finita de posibles
# valores, luego si es necesario explorar ciertas áreas de menor error con valores en ese rango.
# Se entrena el SVR, utilizando un kernel radial.
# También cabe destacar que se utiliza 10-fold cross validation para la validación del modelo.
# Los parámetros a ajustar corresponden a costo, epsilon y gamma y dependiendo de los valores utilizado influirán
# en cuanto se ajusta el modelo a los datos.
# El costo corresponde a una penalización que se le otorga a los puntos de las muestras no separables que se 
# encuentran  en el lado incorrecto del plano, el epsilon define un margen de tolerancia a la penalización
# alrededor de los vectores de soporte y gamma es un parámetro del kernel radial que regula el radio de área de
# influencia de los vectores de soporte.

SVRtune <- tune(svm, Consumo ~.,  data=trainSet,
                   ranges = list(epsilon = seq(0,1,0.1), cost = 2^(-4:9), gamma = 2^(-3:4)), kernel = "radial"
)
print(SVRtune)

# Se realiza la predicción con los mejores parámetros
svrModel <- SVRtune$best.model
svrPrediction <- predict(svrModel, testSet) 
# Se calcula el rmse de la predicción
error <- testSet$Consumo - svrPrediction
rmse(error)


### Random Forest ###
library(randomForest)
library(mlr)
library(ParamHelpers)
# Se define el metodo para realizar 10-fold cross validation
control <- trainControl(method="repeatedcv", number=10, repeats=10)
# Se definen los valores para realizar Grid search en los parámetros que ajustan el modelo para random forest.
# El número de árboles creado (ntree) regula el control del error en el modelo en contraste con el sobreajuste
# del modelo a los datos. También afecta en el tiempo de ejecución el que aumenta a gran escala con un valor muy
# grande. mtry es la cantidad de variables que se seleccionan aleatoriamente para cada división.
# nodesize controla el tamaño mínimo de los nodos terminales y por lo tanto afecta la profundidad del árbol
set.seed(42)
params <- makeParamSet(
  makeDiscreteParam("ntree",values = list(100,300,500,1000)),
  makeDiscreteParam("mtry",values = 3:11),
  makeDiscreteParam("nodesize",values = seq(10,50,10))
)
# Define el método de resampling como validación cruzada
rdesc <- makeResampleDesc("CV",iters=10L)
ctrl <- makeTuneControlRandom(maxit = 10L)
# Define el método de aprendizaje de regresión para random forest
rf.lrn <- makeLearner("regr.randomForest")
# Se indica el conjunto de entrenamiento y la variable predictiva
traintask <- makeRegrTask(data = trainSet,target = "Consumo")
# Se realiza la selección de parámetros
tune <- tuneParams(learner = rf.lrn
                   ,task = traintask
                   ,resampling = rdesc
                   ,par.set = params
                   ,control = ctrl
                   ,show.info = T)

# Se crea el modelo con los mejores parámetros seleccionados
rfModel <- randomForest(Consumo~., data=trainSet, method="rf", trControl=control, mtry=8, nodesize=10,ntree=500)
# Se realiza la predicción en el conjunto de prueba
rfPrediction <- predict(rfModel, testSet) 
# Se obtiene el error RMSE
error <- testSet$Consumo - rfPrediction 
rmse(error)


### Decision trees ###

library(rpart)
# La útlima comparación se realiza con un modelo más sencillo de árboles de desición. Para esto se utiliza
# rpart con el método anova para ajustar la regresión
dtModel <- rpart(Consumo ~ ., data = trainSet,method="anova")
# Se imprime la información del árbol creado
summary(dtModel)
# Se realiza la predicción del conjunto de prueba
dtPrediction <- predict(dtModel, testSet)
dtError <- testSet$Consumo - dtPrediction 
rmse(dtError)


##### Análisis y evaluación de los modelos ### 

# Los errores obtenidos en regresión no son tan fáciles de interpretar como los de clasificación y por lo general
# el criterio de selección se realiza a través de una serie de métricas.
# En primer lugar, se realiza una nueva predicción para el conjunto de test, pero esta vez se ordena para visualizar
# que tan bien se ajusta a la curva en el tiempo

# Se ordena el conjunto de test de forma temporal
testSet <- testSet[order(testSet$Dia),]
testSet <- testSet[order(testSet$Mes),]
testSet <- testSet[order(testSet$A.o),]
# Se realizan las predicciones con los 3 modelos creados
svrPrediction <- predict(svrModel, testSet) 
rfPrediction <- predict(rfModel, testSet) 
dtPrediction <- predict(dtModel, testSet) 
fechaTest <- Fecha[as.integer(rownames(testSet))]

# Se define una función para probar las predicciones de forma visual
plotPrediction <- function(model, tesSet, fechasTest, modelType){
  prediction <- predict(model,testSet)
  
  plot(fechasTest,testSet$Consumo,main="Original vs Prediction",xlab="Fecha", ylab="Consumo"
       ,col = "blue", type = "l")
  lines(fechaTest,prediction, col="red", type = "l")
  legend("topleft", legend=c("Original", modelType),
         col=c("blue", "red"), lty=1, cex=0.8)
}
# Predicciones SVR
plotPrediction(svrModel,testSet,fechaTest,"SVR")
# Predicciones Random Forest
plotPrediction(rfModel,testSet,fechaTest,"Random Forest")
# Predicciones Decision Tree
plotPrediction(dtModel,testSet,fechaTest,"Decision Tree")

# Esta es una de las formas más interesantes de porbar el modelo, ya que independiente del error inherente en la
# predicción, se puede observar que tan similar son los datos en contraste con el conjunto real. En este caso
# las predicciones resultan bastante acertadas a lo largo de los años, el modelo que se ajusta de forma menos
# exacta es el de decision trees. Esto era predecible ya que es un modelo simple que no requiere de mucho ajuste
# pero aún así obtiene resultados cercanos y el tiempo de entrenamiento es mucho más bajo que los otros modelos, 
# prácticamente se obtiene al instante.

# Ahora se evalúan los modelos en base a distintas métricas de error que usualmente son aplicadas a las predicciones
# de regresión. Las medidas para evaluar los modelos son las siguientes:
# MSE (Mean Squared Error): es el promedio de la diferencia entre los valores originales y los de la predicción.
# MAE (Mean Absolute Error): es el promedio de la diferencia en valor absoluto entre el conjunto otriginal y la 
# predicción.
# R^2 score: Mide la calidad del modelo para replicar resultados y la proporción de variación de los resultados 
# que puede explicarse por el modelo
# RMSE (Root Mean Squared Error): Es una medida de qué tan dispersos o concentrados están los residuos de la línea
# original o de ajuste
R2 <- function(original,predict){
  1 - (sum((original-predict )^2)/sum((original-mean(original))^2))
}
library(MLmetrics)
errors <- function(svrModel, rfModel, dtModel, testSet){
  svrPrediction = predict(svrModel,testSet)
  rfPrediction = predict(rfModel,testSet)
  dtPrediction = predict(dtModel,testSet)
  # MSE
  cat("\t  SVR \t-\t RF \t-\t DT\n")
  cat("MSE:\t",MSE(svrPrediction,testSet$Consumo)," - ",
  MSE(rfPrediction,testSet$Consumo)," - ",
  MSE(dtPrediction,testSet$Consumo),"\n")
  # MAE
  cat("MAE:\t",MAE(svrPrediction,testSet$Consumo)," - ",
      MAE(rfPrediction,testSet$Consumo)," - ",
      MAE(dtPrediction,testSet$Consumo),"\n")
  # R^2
  cat("R^2:\t",R2(svrPrediction,testSet$Consumo)," - ",
      R2(rfPrediction,testSet$Consumo)," - ",
      R2(dtPrediction,testSet$Consumo),"\n")
  # RMSE
  cat("RMSE:\t",rmse(svrPrediction-testSet$Consumo)," - ",
      rmse(rfPrediction-testSet$Consumo)," - ",
      rmse(dtPrediction-testSet$Consumo),"\n")
}

errors(svrModel,rfModel,dtModel,testSet)

# En los errores presentados se puede notar que en general el modelo de Random Forest fue el que obtuvo las mejores
# predicciones y Decision Trees las peores. El caso de estudiar varios errores es debido a que en ocasiones los
# valores son tan cercanos que se necesita otra medida, por ejemplo si el MAE es igual, el MSE indica cual
# posee mayor dispersión, o la calidad con el estimador R^2 (cuyo valor máximo es 1). En este caso los errores
# resultan claramente diferenciables.

# El siguiente ejercicio es predecir valores en distintos rangos de tiempo. Para esto se toma un pequeño conjunto
# de datos para tomar algunas medidas

testSet2 <- gasConsume[(nrow(gasConsume)*0.9):nrow(gasConsume),]
testSet2$fecha <- Fecha[as.integer(rownames(testSet2))]
dmy <- dummyVars(" ~ .", data = testSet2)
testNumeric <- data.frame(predict(dmy, newdata = testSet2))
testNumeric$Dia <- range01(testNumeric$Dia)
testNumeric$Mes <- range01(testNumeric$Mes)
head(testNumeric)
rfModel <- randomForest(Consumo~., data=testNumeric, method="rf", trControl=control, mtry=8, nodesize=10,ntree=500)
testNumeric$predictions <- predict(rfModel, testNumeric) 
testNumeric$fecha <- Fecha[as.integer(rownames(testNumeric))]

# Con una función se pueden desplegar gráficos para mostrar la predicción en distintos rangos
predictions <- function(data,startDate,endDate){
  predictions <- subset(data$predictions,data$fecha >= as.Date(startDate) & data$fecha <= as.Date(endDate))
  date <- subset(data$fecha,data$fecha >= as.Date(startDate) & data$fecha <= as.Date(endDate))
  plot(date,predictions,main="Predicciones",xlab="Fecha", ylab="Consumo"
       ,col = "blue", type = "l")
}
# Una semana
predictions(testNumeric,"2002-04-12","2002-04-18")
# Un mes
predictions(testNumeric,"2002-04-12","2002-05-12")
# Tres meses
predictions(testNumeric,"2002-04-12","2002-06-12")
# Cinco meses
predictions(testNumeric,"2002-04-12","2002-08-18")
testSet2$date <- Fecha[(nrow(gasConsume)*0.9):nrow(gasConsume)]

# Para entregar resultados de mayor impacto para el cliente se pueden usar bibliotecas como plot_ly que permiten
# interactividad con los datos. En este ejemplo se incluyen sliders para acotar el conjunto de predicciones
# y botones en la parte superior para fijar ciertas zonas. Además se muestran los valores del consumo al deslizar
# el mouse por arriba de los valores
library(plotly)
plot_ly(testSet2, x = ~fecha) %>%
  add_lines(y = ~Consumo) %>%
  layout(
    title = "",
    xaxis = list(
      rangeselector = list(
        buttons = list(
          list(
            count = 1,
            label = "1 mes",
            step = "Mes",
            stepmode = "backward"),
          list(
            count = 3,
            label = "3 meses",
            step = "Mes",
            stepmode = "backward"),
          list(
            count = 5,
            label = "5 meses",
            step = "Mes",
            stepmode = "backward"),
          list(step = "all"))),
      
      rangeslider = list(type = "date")),
    
    yaxis = list(title = "Consumo"))

## Como conclusión final sólo queda agregar que la efectividad de los modelos puede aumentar en gran medida
## al agregar más variables y muetras a los datos. En general en comparación SVR y random forest muchas veces
## logran buenos resultados con un buen ajuste de parámetros, el cual se puede realizar de forma iterativa
## infinitas veces pero todo depende del tiempo dispuesto para esta tarea en contraste con la tolerancia al error
## de los resultados. Al ser un conjunto de datos pequeño, probablemente al utilizar random forest con los parámetros
## por defecto se hubiera llegado a un error similar, en cambio SVR podría tener una mejor oportunidad de 
## crear el mejor modelo en un conjunto con muchas más variables.
