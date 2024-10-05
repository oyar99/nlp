from re import compile
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, MinMaxScaler
from pandas import DataFrame, concat



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


def build_preprocess_pipeline(vectorizer_type: str = 'count', max_features: int = 15000) -> Pipeline:
    """
    Construye un pipeline de procesamiento y vectorización de texto utilizando scikit-learn.

    El pipeline incluye:
    - Limpieza de texto usando `clean_text` para normalizar los datos.
    - Vectorización del texto utilizando `CountVectorizer` o `TfidfVectorizer` para convertir el texto en una matriz de características.
    - Si se selecciona 'count' como vectorizador, se aplicará `MaxAbsScaler` para escalar las características, manteniendo las matrices dispersas.

    Args:
        vectorizer_type (str, optional): Tipo de vectorizador a utilizar. 
            Puede ser 'count' para `CountVectorizer` o 'tfidf' para `TfidfVectorizer`. 
            El valor predeterminado es 'count'.
        max_features (int, optional): Número máximo de características que se utilizarán 
            en el vectorizador. Este parámetro controla la dimensionalidad del espacio 
            de características. El valor predeterminado es 15000.

    Raises:
        ValueError: Si `vectorizer_type` no es 'count' ni 'tfidf'.

    Returns:
        Pipeline: Un pipeline de scikit-learn listo para preprocesar y vectorizar 
        datos de texto.
    """
    # Selección del vectorizador según el tipo especificado
    if vectorizer_type == 'count':
        vectorizer = CountVectorizer(
            max_features=max_features,      # Limita el número de características
            stop_words='english',           # Elimina las palabras vacías (stopwords) en inglés
            ngram_range=(1, 2)              # Incluye unigrams y bigrams
        )
    elif vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=max_features,      # Limita el número de características
            stop_words='english',           # Elimina las palabras vacías (stopwords) en inglés
            ngram_range=(1, 2)              # Incluye unigrams y bigrams
        )
    else:
        # Si el tipo de vectorizador no es válido, se lanza un error
        raise ValueError("El parámetro 'vectorizer_type' debe ser 'count' o 'tfidf'.")

    # Definición de los pasos del pipeline
    steps = [
        # Limpieza de texto utilizando una función personalizada
        ('clean_text', FunctionTransformer(lambda x: [clean_text(text) for text in x], validate=False)),
        # Vectorización del texto con el vectorizador seleccionado
        ('vectorizer', vectorizer),
    ]
    
    # Añadir escalado si se utiliza CountVectorizer (las matrices dispersas pueden beneficiarse del escalado)
    if vectorizer_type == 'count':
        steps.append(('scaler', MaxAbsScaler()))  # Escala las características al rango [-1, 1]

    # Construcción y retorno del pipeline
    pipeline = Pipeline(steps)
    return pipeline


def create_sentiment_dataset(files: list[str]) -> DataFrame:
    """
    Crea un dataset de texto y etiquetas a partir de una lista de archivos.

    Esta función lee múltiples archivos de texto, extrae etiquetas de sentimiento
    (basadas en un patrón específico) y realiza preprocesamiento del texto, como 
    la eliminación de patrones no deseados. El resultado es un DataFrame que contiene 
    el texto preprocesado, las etiquetas y la información sobre la fuente de los archivos.

    Args:
        files (list[str]): Lista de rutas a los archivos de texto que contienen los datos.

    Returns:
        DataFrame: Un DataFrame que contiene las siguientes columnas:
            - 'raw_text': El texto original leído del archivo.
            - 'label': La etiqueta de sentimiento extraída del texto.
            - 'text': El texto preprocesado donde se eliminan ciertos patrones.
            - 'folder': El nombre de la carpeta de origen del archivo.
            - 'file': El nombre del archivo de origen.
    """

    # Compilar las expresiones regulares para extraer las etiquetas y limpiar el texto
    label_match = compile(r'#label#:(.*)')  # Patrón para extraer la etiqueta de sentimiento
    combined_pattern = compile(r'#label#:(.*)|:\d+|_')  # Patrón para limpiar el texto

    # Lista para acumular los DataFrames resultantes de cada archivo
    dfs = []
    
    # Iterar sobre cada archivo en la lista de archivos proporcionados
    for path in files:
        # Leer el contenido del archivo y dividirlo en líneas de texto
        data = read_file(path).split('\n')
        
        # Extraer el nombre de la carpeta y del archivo a partir de la ruta
        folder, file = path.replace('\\', '/').split('/')[-2:]

        # Crear un DataFrame a partir del texto leído
        df = DataFrame({'raw_text': data})
        
        # Filtrar las filas donde el texto tiene al menos 5 caracteres
        df = df[df['raw_text'].str.len() > 5]

        # Extraer la etiqueta de sentimiento usando la expresión regular (vectorizado)
        df['label'] = df['raw_text'].str.extract(label_match, expand=False)

        # Limpiar el texto eliminando patrones no deseados (vectorizado)
        df['text'] = df['raw_text'].str.replace(combined_pattern, ' ', regex=True)

        # Añadir las columnas de información de la carpeta y archivo de origen
        df['folder'] = folder
        df['file'] = file
        
        # Agregar el DataFrame a la lista de DataFrames
        dfs.append(df)
    
    # Concatenar todos los DataFrames acumulados en uno solo y reiniciar el índice
    df_final = concat(dfs).reset_index(drop=True)
    
    return df_final

def sentiment_score(text: str, lexicon: dict):
    """
    Calcula la puntuación de sentimiento de un texto

    Args:
        text: El texto de entrada
    """
    tokens = text.split(' ')
    score = 0
    for token in tokens:
        if token in lexicon:
            pos_score, neg_score = lexicon[token]
            score += pos_score - neg_score
    return score

def build_preprocess_pipeline_lexicon(path: str) -> Pipeline:
    """
    Construye un pipeline de procesamiento y vectorización de texto utilizando scikit-learn.

    El pipeline incluye:
    - Limpieza de texto usando `clean_text` para normalizar los datos.
    - Extraccion de caracteristicas utilizando un lexicon

    Returns:
        Pipeline: Un pipeline de scikit-learn listo para preprocesar y vectorizar 
        datos de texto.
    """
    lexicon = load_lexicon(path)
    
    # Definición de los pasos del pipeline
    steps = [
        # Limpieza de texto utilizando una función personalizada
        ('clean_text', FunctionTransformer(lambda x: [clean_text(text) for text in x], validate=False)),
        # Calculo de puntajes de sentimiento
        ('sentiment_score', FunctionTransformer(lambda x: [[sentiment_score(text, lexicon)] for text in x],
                                                validate=False)),
        # Normaliza los resultados
        ('scaler', MinMaxScaler()),
    ]
    
    # Construcción y retorno del pipeline
    pipeline = Pipeline(steps)
    return pipeline

def load_lexicon(path: str) -> dict:
    """
    Carga un lexicon para analisis de sentimientos y devuelve un diccionario
    donde cada llave corresponde a un termino del lexicon y el valor a una tupla
    con el score positivo y negativo indicando si es un termino que se usa en un
    contexto de una resena buena o mala respectivamente

    Args:
        path: str: La ruta donde esta almacenada el lexicon de SentiWordNet

    Returns:
        dict: Un diccionario con los puntajes positivo/negativo de cada termino
    """
    lexicon = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('#'):
                parts = line.split('\t')
                pos_score = float(parts[2]) # Score positivo
                neg_score = float(parts[3]) # Score negativo
                terms = parts[4].split()
                
                # Almacenar la puntuacion de cada termino en el diccionario
                for term in terms:
                    # Limpiar el termino para eliminar hashtags y números que indican
                    # posibles significados de la palabra pero por simplicidad se ignoran
                    term_clean = term.split('#')[0].strip()
                    lexicon[term_clean] = (pos_score, neg_score)

    return lexicon

