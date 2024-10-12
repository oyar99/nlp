# Embeddings de Texto y Redes FeedForward

Este repositorio contiene varios notebooks de python demostrando el uso de embeddings de texto y redes feedforward
para entrenar clasificadores de texto.

## Datos

Se descargaron varios libros en texto plano de [Gutenberg](https://www.gutenberg.org/) de 3 autores, `Jane Austen`, `Leo Tolstoy`, y `James Joyce`. Estos se encuentran en la ruta `data/raw`. Se entrenaron embeddings para las palabras del vocabulario de estos libros, y estos se almacenaron en `data/models`.

## Notebooks

Exite un notebook para cada uno de los puntos de la actividad.

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
pip install -r requirements.txt
```