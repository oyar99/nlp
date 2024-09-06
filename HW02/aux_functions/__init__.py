from re import compile

# Compilamos las expresiones regulares fuera de la función para mejorar la eficiencia
email_re = compile(r'\S*@\S*\s?|from: |re: |subject: |urllink|maxaxaxaxaxaxaxaxaxaxaxaxaxaxax')
punctuation_re = compile(r'[?!:]')
non_alphanumeric_re = compile(r'[^A-Za-z0-9. \n]')
numbers_re = compile(r'\b\d+\b')
multiple_newlines_re = compile(r'\n{2,}')
single_newline_re = compile(r'\n')
multiple_dots_re = compile(r'\.\.+')
multiple_spaces_re = compile(r'\s+')
multiple_char_repetition_re = compile(r'(.)\1{2,}')

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
    
    # Convertir el texto a minúsculas
    txt = txt.lower()

    # Eliminar correos electrónicos, etiquetas dentro de <>, y ciertas palabras clave
    txt = email_re.sub('', txt)

    # Reemplazar secuencias de puntos múltiples con un solo espacio
    txt = multiple_dots_re.sub(' ', txt)

    # Eliminar repeticiones de caracteres consecutivos (más de dos veces)
    txt = multiple_char_repetition_re.sub(r'\1', txt)

    # Reemplazar guiones por espacios
    txt = txt.replace('-', ' ')

    # Sustituir signos de puntuación específicos por un punto
    txt = punctuation_re.sub('.', txt)

    # Eliminar caracteres no alfanuméricos excepto puntos, espacios y saltos de línea
    txt = non_alphanumeric_re.sub('', txt)

    # Sustituir números por 'NUM'
    txt = numbers_re.sub('NUM', txt)

    # Reemplazar múltiples saltos de línea consecutivos por un solo punto
    txt = multiple_newlines_re.sub('.', txt)

    # Convertir saltos de línea individuales en espacios
    txt = single_newline_re.sub(' ', txt)

    # Reemplazar múltiples espacios consecutivos por un solo espacio
    txt = multiple_spaces_re.sub(' ', txt)

    # Eliminar espacios en blanco al inicio y final del texto
    return txt.strip()
