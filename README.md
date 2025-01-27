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

A continuación se describen los principales hallazgos encontrados en el análisis exploratorio

*Insights*

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
<li> <b> Correlación </b> </li>Hay una alta incidencia de correlación entre las variables de las mismas familias. Donde para las variables con menos del 20% de valores nulos se tienen las siguientes incidencias entre variables con los mismos nombres base. 

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

</ul>

<a names= "Preprocesamiento"></a>

## Preprocesamiento

El preprocesamiento se dividió en 3 secciones con una condición inicial sobre los datos, dicha condición consiste en tratar por separado (trabajo a futuro) los registros con alta nulicidad. Por lo que una cantidad de registros baja se separó para su propio estudio posterior.

Las secciones en orden de implementación son "separación de conjuntos", "selección de variables" y "transformaciones". A continuación se detalla el procedimiento.

<ul>
<li> Validate databricks inputs </li>
<li> Transform tables and dataframes </li>
<li> Download and upload data from/to external sources as MongoDB, Mysql, AWS and Elastic search</li>
<li> Automate data persistence within databricks (performs CRUD operations)</li>
</ul>

<a names= "Modelación"></a>

## Modelación


[//]: <> (This is also a comment.)

<a names= "Resultados"></a>

## Resultados




