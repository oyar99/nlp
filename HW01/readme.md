# Recuperación de Información

Este repositorio contiene varios notebooks de python para resolver e implementar tareas comunes de recuperación de información.

## Datos

Los datos con los que se trabaja han sido cargados en la carpeta de `data` dentro de `src`.

- `Docs raw texts` contiene 331 documentos en formato NAF.
- `Queries raw texts` contiene 35 consultas.
- `relevance-judgements.tsv` contiene para cada consulta los documentos considerados relevantas para cada una de las consultas.

## 1-Metricas De Evaluacion de IR

Contiene implementaciones usando numpy de metricas comunes de evaluacion de IR.

## 2-Busqueda Binaria Usando Indice Invertido

Contiene una implementacion de recuperacion de informacion usando un indice invertido usando estructuras nativas de Python.

## 3-Recuperacion Ranqueada y Vectorizacion de Documentos

Contiene una implementacion de recuperacion ranqueada

## 4-Recuperacion Ranqueada (GENSIM)

Contiene una implementacion de recuperacion ranquada usando la libreria de GESIM

## Como ejecutar?

Utilizar Python 3.9

```sh
python --version
```

Crear un ambiente virtual llamado `env`

```sh
python -m venv env
```

Activar el ambiente virtual

```sh
.\env\Scripts\activate
```

Instalar las librerias requeridas

```sh
pip install -r src/requirements.txt
```