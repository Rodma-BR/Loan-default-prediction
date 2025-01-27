# Loan Default Prediction

<h1 align="center">
  <img src="./figs/money.jpeg" width = 600 alt="Openspark logo">
  <br>
</h1>

### Table of Contents  

[Resumen](#Resumen) <br>
[Análisis](#EDA) <br>
[Preprocesamiento](#Preprocesamiento) <br>
[Modelación](#Modelación) <br>
[Resultados](#Resultados) <br>
[Trabajo_futuro](#Trabajo_futuro) <br>

<a names= "Resumen"></a>

## Resumen

En este proyecto se implemetó modelo de regresión logística entrenado para la detección de incumplimiento de un préstamo, siendo este un problema de clasificación binaria. 
Además de los parámetros del modelo, se tienen organizados y distribuidos los recursos para el análisis exploratorio (EDA), preprocesamiento y entrenamiento. Los detalle de cada sección se describen con detalle en este mismo texto.

<a names= "EDA"></a>

## Análisis explotario de datos (EDA)

En el análisis exploratorio se revisaron con cuidado los registros del conjunto de datos "definitely_not_from_kaggle_loan_default_dataset.csv". La principales tareas de análisis se pueden agrupar en la serie de pasos siguientes.

<ol>
<li> Revisión de las características (Tipo de dato, nombres, similitudes) </li>
<li> Revisión del porcentaje de valores nulos por variable </li>
<li> Revisión de la completez de la información por registro </li>
<li> Correlación entre las variables </li>
</ol>

También se revisan estadísticos sobre las variables, distribución de los datos, manejo de valores atípicos, entre otros procedimientos. En el notebook EDA.ipynb se encuentran la exploración más importante y se eliminó el código que no llevó a ningún descubrimiento relevante.

---
> [!NOTE] 
> El conjunto de datos no cuenta con un diccionario por lo que se desconoce el origen y la importancia a priori de las variables a analizar. El conocimiento inicial permite un mejor estudio de las variables y la posibilidad de desarrollar nuevas variables por medio de ingeniería.

---

A continuación se describen los principales hallazgos encontrados en el análisis exploratorio, estos se resumen en la siguiente lista y se describen posteriormente.

<ul>
<li> Clases: </li> 
<li> Registros nulos: </li>
<li> Familias de variables: </li> 
<li> Variables seudocontinuas </li>
<li>  Correlación </li>
<li> Outliers </li>
<li> Distribución </li>
<li> Regiones </li> 
<li> Distribución de incumplimiento </li>
</ul>

**Insights**

<ul>
<li> <b>Clases: </b></li> Es un problema binario con clases desbalanceadas, la proporción es como sigue

|Default| Porcentaje|
|--|--|
|1| 17%|
|0| 83%|

<li> <b>Registros nulos: </b></li> Existen registros con más del 90% de valores nulos, esto es un problema respecto a la calidad del dato. Se requiere un estudio más minucioso sobre este subcojunto.
<li> <b>Familias de variables: </b> </li> Existen familias de variables con varios tipos principales, algunas con métricas agregadas (min, max, mean, no agg). De estas familias se encuentra que la mayor es la relacionada al nombre "activity_pattern" con 104 variables similares. 
<h1 align="center">
  <img src="./figs/similar_features.png" width = 600>
  <br>
</h1>
<li> <b>Variables seudocontinuas</b> </li> Se encontró que un proporción considerable de variables supuestamente continuas tienen valores fijos y con una baja cantidad de únicos. Por ejemplo, "activity_pattern_09_last_9_months_max" que cuenta solo con 23 valores únicos.
<li> <b> Tasa de valores faltantes </b> </li>Aproximadamente la mitad de las variables presentan un porcentaje mayor al 20% de valores nulos.
<h1 align="center">
  <img src="./figs/completeness.png" width = 600>
  <br>
</h1>
<li> <b> Correlación </b> </li> Hay una alta incidencia de correlación entre las variables de las mismas familias. Donde para las variables con menos del 20% de valores nulos se tienen las siguientes incidencias entre variables con los mismos nombres base. 

```python
{'mobility_pattern_work_': 5,
 'activity_pattern_': 249,
 'device_inferred_price_': 1,
 'social_mean_degree_last_': 3,
 'entropy_contacts_weekend_last_': 1}
```
<li> <b> Outliers </b> </li> Existe una presencia alta de valores atípicos en varias variables. A continuación se muestra una ejemplo para variables agregadas por la media aritmética

<h1 align="center">
  <img src="./figs/outliers.png" width = 300 alt = "outliers">
  <br>
</h1>

<li> <b> Distribución </b> </li> Los datos presentan distribuciones por bloques, mostrando que los datos no se distribuyen continuamente sobre todo el dominio. Se pueden apreciar este tipo de aglomeraciones en las variables "activity_pattern"

<h1 align="center">
  <img src="./figs/activity_pattern_pairs.png" width = 600 alt = "outliers">
  <br>
</h1>


<li><b> Regiones </b> </li> Se encontró que hay un porcentaje similar de incumplimientos por región, siendo entonces que no existe una diferenciación clara por geografía respecto al incumplimiento del pago del préstamo.

|Región|default|
|--|--|
|6|0.112782|
|8|0.149629|
|4|0.12987|
|7|0.141616|
|3|	0.12311|
|1|0.134247|
|2|0.146296|
|5|0.129139|
|9|0.130252|

<li> <b> Distribución de incumplimiento </b> </li> El incumplimiento no se distribuye uniformemente sobre el tiempo, siendo particularmente notorio en el último mes del conjunto etiquetado.

<h1 align="center">
  <img src="./figs/default_time_series.png" width = 600 alt = "default_ts">
  <br>
</h1>
</ul>

<a names= "Preprocesamiento"></a>

## Preprocesamiento

El preprocesamiento se dividió en 3 secciones con una condición inicial sobre los datos, dicha condición consiste en tratar por separado (trabajo a futuro) los registros con alta nulicidad. Por lo que una cantidad de registros baja se separó para su propio estudio posterior.

Las secciones en orden de implementación son,

<ul>
<li> Separación de conjuntos </li>
<li> Selección de variables </li>
<li> Transformaciones</li>
</ul>

A continuación se detalla el procedimiento,

<ul>
<li> <b> Separación de conjuntos: </b> </li> El conjunto de entrenamiento y validación se separó considerando la dependencia temporal de las predicciones sobre el conjunto de prueba (muestras sin etiqueta). Considerando esta necesidad, se separaron 10 meses para el conjunto de entrenamiento y 4 meses para el conjunto de validación. 

La distribución de incumplimiento para ambos conjuntos es similar

**Conjunto de entrenamiento**

|default| Proporción de la clase|
|--|--|
|0| 0.839722 |
|1|0.160278 |


**Conjunto de validación**

|default| Proporción de la clase|
|--|--|
|0| 0.80965  |
|1|0.19035 |

La distribución del total de registros entre conjuntos es aproximadamente del 70%-30% para entrenamiento y validación respectivamente.

<li> <b> Selección de variables: </b> </li> Se realizaron procedimientos para la exclusión de variables según las siguientes condiciones.
<ul> 
<li> Variables no deseadas: </li> Se excluyeron varias que no son deseadas desde el inicio del estudio como el id del registro o la fecha en la que se registró el crédito, esta última para evitar la dependecia del modelo respecto al tiempo.
<li> Variables quasi-constantes: </li> Las variables con baja varianza, generalmente, tienen poco impacto en la implementación de los modelos ML.
<li> Variables con alto porcentaje de valores nulos: </li> Se retiraron las variables que presentaban una proporción mayor al 20% de valores nulos, considerando que restaban más de la mitad de las variables haciendo dicha exclusión.
<li> Variables correlacionadas: </li> Se excluyeron las variables con una correlación mayor al 0.8 según el método clásico de Pearson. Si hay correlación en una relación uno a muchos se retiene la primer variable encontrada. Más información del método se puede consultar el la documentación de <a href = "https://feature-engine.trainindata.com/en/latest/index.html">Feature engine</a>.
</ul>
<li> <b> Creación de variables: </b> </li> Este procedimiento no se realizó en este estudio por falta de definición de las variables.
<li><b>  Transformaciones: </b> </li> Se realizaron 2 transformaciones principales.
<ul> 
<li> One hot encoder: </li> La variable de regiones se transformó utilizando OneHotEncoder por si el modelo selecciona alguna de estas regiones siguiendo el algoritmo RFE (Se revisa en la implementación)
<li> Imputación: </li> Se imputaron los valores faltantes para las variables seleccionadas siguiendo el algoritmo de KNN, la razón principal fue la no continuidad de los valores de las variables y la presencia de aglomeraciones de datos.
</ul>
</ul>

Finalmente se preservan los conjuntos de datos limpios para la posterior implementación del modelo.

<a names= "Modelación"></a>

## Modelación


[//]: <> (This is also a comment.)

<a names= "Resultados"></a>

## Resultados

Mejora a la selección de variables correlacionadas


