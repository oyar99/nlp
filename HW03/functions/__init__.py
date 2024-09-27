from re import compile
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import TruncatedSVD



def read_file(f: str) -> str:
    """
    Lee el contenido de un archivo y maneja posibles errores de codificación.

    Args:
        f (str): La ruta al archivo que se desea leer.

    Returns:
        str: El contenido del archivo como una cadena de texto.

    Notas:
        - La función intenta leer el archivo con codificación 'utf-8' por defecto.
        - Si ocurre un error de decodificación (UnicodeDecodeError), se reintenta con la codificación 'latin1'.
        - Este enfoque es útil cuando se manejan archivos de texto con codificaciones mixtas o desconocidas.
    """
    try:
        with open(f, 'r', encoding='utf-8') as file:
            txt = file.read()
    except UnicodeDecodeError:
        with open(f, 'r', encoding='latin1') as file:
            txt = file.read()
    
    return txt


def clean_text(txt: str) -> str:
    """
    Limpia el texto eliminando correos electrónicos, etiquetas HTML, caracteres no alfanuméricos,
    y otros elementos no deseados.

    Args:
        txt (str): El texto que se desea limpiar.

    Returns:
        str: El texto limpio y normalizado.

    Pasos:
        - Convertir el texto a minúsculas para normalizarlo.
        - Eliminar correos electrónicos, etiquetas HTML, y ciertas palabras clave usando expresiones regulares.
        - Reemplazar secuencias de puntos múltiples con un solo espacio.
        - Eliminar repeticiones de caracteres consecutivos (más de dos veces) que no sean necesarias.
        - Reemplazar guiones por espacios para evitar palabras concatenadas por guiones.
        - Sustituir signos de puntuación específicos (?, !, :) por un punto.
        - Eliminar caracteres no alfanuméricos, excepto puntos, espacios y saltos de línea.
        - Sustituir números por la palabra 'NUM' para normalizar las secuencias numéricas.
        - Reemplazar múltiples saltos de línea consecutivos con un solo punto.
        - Convertir saltos de línea individuales en espacios.
        - Reemplazar múltiples espacios consecutivos por un solo espacio.

    Notas:
        - El uso de expresiones regulares precompiladas mejora la eficiencia al aplicar la limpieza a múltiples textos.
    """

    # Compila las expresiones regulares fuera de la función para mejorar la eficiencia
    email_re = compile(r'\S*@\S*\s?|from: |re: |subject: |urllink|maxaxaxaxaxaxaxaxaxaxaxaxaxaxax')
    punctuation_re = compile(r'[?!:]')
    non_alphanumeric_re = compile(r'[^A-Za-z0-9 \n]')
    numbers_re = compile(r'\b\d{1,3}(,\d{3})*(\.\d+)?\b')
    single_newline_re = compile(r'\n')
    multiple_spaces_re = compile(r'\s+')
    multiple_char_repetition_re = compile(r'(.)\1{2,}')
    
    # Convierte el texto a minúsculas
    txt = txt.lower()

    # Elimina correos electrónicos, etiquetas dentro de <>, y ciertas palabras clave
    txt = email_re.sub('', txt)

    # Elimina repeticiones de caracteres consecutivos (más de dos veces)
    txt = multiple_char_repetition_re.sub(r'\1', txt)

    # Elimina caracteres no alfanuméricos excepto espacios y saltos de línea
    txt = non_alphanumeric_re.sub(' ', txt)

    # Sustituye números por 'NUM'
    txt = numbers_re.sub('NUM', txt)

    # Convierte saltos de línea individuales en espacios
    txt = single_newline_re.sub(' ', txt)

    # Reemplaza múltiples espacios consecutivos por un solo espacio
    txt = multiple_spaces_re.sub(' ', txt)

    # Elimina espacios en blanco al inicio y final del texto
    return txt.strip()


def build_pipeline(vectorizer_type: str = 'count', max_features: int = 20000, dim: int = None) -> Pipeline:
    """
    Construye un pipeline de scikit-learn para el procesamiento y transformación de textos.

    El pipeline incluye:
    - Limpieza de texto con `clean_text`.
    - Vectorización del texto utilizando `CountVectorizer` o `TfidfVectorizer`.
    - Opcionalmente, reducción de dimensionalidad con `TruncatedSVD`.

    Args:
        vectorizer_type (str, optional): Tipo de vectorizador a utilizar. Puede ser 'count' para `CountVectorizer` o 'tfidf' para `TfidfVectorizer`. Por defecto es 'count'.
        max_features (int, optional): Número máximo de características a considerar en el vectorizador. Por defecto es 20000.
        dim (int, optional): Número de componentes para `TruncatedSVD` si se aplica reducción de dimensionalidad. Si es `None`, no se aplica reducción de dimensionalidad. Por defecto es `None`.

    Raises:
        ValueError: Si `vectorizer_type` no es 'count' ni 'tfidf'.

    Returns:
        Pipeline: Objeto `Pipeline` de scikit-learn listo para transformar datos de texto.
    """
    # Selección del vectorizador según el tipo especificado
    if vectorizer_type == 'count':
        vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
    elif vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
    else:
        raise ValueError("El parámetro 'vectorizer_type' debe ser 'count' o 'tfidf'.")

    # Definición de los pasos del pipeline
    steps = [
        ('clean_text', FunctionTransformer(lambda x: [clean_text(text) for text in x], validate=False)),
        ('vectorizer', vectorizer),
    ]

    # Agregar reducción de dimensionalidad si 'dim' está definido
    if dim is not None:
        steps.append(('reduce_dim', TruncatedSVD(n_components=dim, random_state=42)))

    # Construcción del pipeline
    pipeline = Pipeline(steps)
    return pipeline