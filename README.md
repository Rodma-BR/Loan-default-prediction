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
<li> Revisión de la completez de las variables </li>
<li> Revisión de completez de los registros </li>
<li> Correlación entre las variables </li>
</ol>

También se revisan estadísticos sobre las variables, distribución de los datos, manejo de valores atípicos, entre otros procedimientos. En el notebook EDA.ipynb se encuentran la exploración más importante y se eliminó el código que no llevó a ningún descubrimiento relevante.

---
<span style="color:red"> *Nota:* </span> <br>
  El conjunto de datos no cuenta con un diccionario por lo que se desconoce el origen y la importancia a priori de las variables a analizar. El conocimiento inicial permite un mejor estudio de las variables y la posibilidad de desarrollar nuevas variables por medio de ingeniería.

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
  <img src="./figs/similar_features.png" width = 600 alt="Openspark logo">
  <br>
</h1>
<li> <b>Variables seudocontinuas</b> </li> Se encontró que un proporción considerable de variables supuestamente continuas tienen valores fijos y con una baja cantidad de únicos. Por ejemplo, "activity_pattern_09_last_9_months_max" que cuenta solo con 23 valores únicos.

<li> Aproximadamente la mitad de las variables presentan un porcentaje mayor al 20% de valores nulos. </li>
<li>  Hay una alta incidencia de correlación entre las variables de las mismas familias. Donde para las variables con menos del 20% de valores nulos se tienen las siguientes incidencias entre variables con los mismos nombres base.  </li>

```python
{'mobility_pattern_work_': 5,
 'activity_pattern_': 249,
 'device_inferred_price_': 1,
 'social_mean_degree_last_': 3,
 'entropy_contacts_weekend_last_': 1}
```
</ul>
<a names= "features"></a>

## Main features

<ul>
<li> Validate databricks inputs </li>
<li> Transform tables and dataframes </li>
<li> Download and upload data from/to external sources as MongoDB, Mysql, AWS and Elastic search</li>
<li> Automate data persistence within databricks (performs CRUD operations)</li>
</ul>

<a names= "setup"></a>

## Setup

### Python

Install the latest Openspark version with:
!!! It is not available yet!!!

sh
pip install -i https://test.pypi.org/simple/ Openspark


[//]: <> (This is also a comment.)

<a names= "overview"></a>

## Examples

### Module

python
>>> # Verify if date is in the specified format
>>> validate_date("2024-01-01", format_ = "%Y-%m-%d")
"Valid date"
True


<p align="center">
  <b>Documentation (Pending)</b>:
  <a href="">Python</a>
</p>



[![PyPI version](https://img.shields.io/pypi/v/openai.svg)](https://pypi.org/project/openai/)

The OpenAI Python library provides convenient access to the OpenAI REST API from any Python 3.8+
application. The library includes type definitions for all request params and response fields,
and offers both synchronous and asynchronous clients powered by [httpx](https://github.com/encode/httpx).

It is generated from our [OpenAPI specification](https://github.com/openai/openai-openapi) with [Stainless](https://stainlessapi.com/).

## Documentation

The REST API documentation can be found on [platform.openai.com](https://platform.openai.com/docs). The full API of this library can be found in [api.md](api.md).

## Installation

> [!IMPORTANT]
> The SDK was rewritten in v1, which was released November 6th 2023. See the [v1 migration guide](https://github.com/openai/openai-python/discussions/742), which includes scripts to automatically update your code.

```sh
# install from PyPI
pip install openai
```

## Usage

The full API of this library can be found in [api.md](api.md).

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-4o",
)
```

While you can provide an `api_key` keyword argument,
we recommend using [python-dotenv](https://pypi.org/project/python-dotenv/)
to add `OPENAI_API_KEY="My API Key"` to your `.env` file
so that your API Key is not stored in source control.

