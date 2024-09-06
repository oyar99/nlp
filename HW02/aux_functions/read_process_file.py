from . import clean_text
from re import findall, split, DOTALL

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

def extract_and_process_text_from_xml(file_path: str) -> list:
    """
    Extrae y procesa el texto contenido entre etiquetas <post> en un archivo XML.

    Args:
        file_path (str): La ruta al archivo XML del cual se extraerá el texto.

    Returns:
        list: Una lista de diccionarios, donde cada diccionario representa una oración extraída y procesada
              con las claves:
              - 'text': La oración procesada.
              - 'length': La longitud de la oración en número de palabras.

    Notas:
        - La función primero lee el archivo XML utilizando `read_file`.
        - Luego extrae el contenido de todas las etiquetas <post> usando expresiones regulares.
        - Cada fragmento de texto extraído se limpia utilizando la función `clean_text`.
        - El texto limpio se divide en oraciones, que se formatean y almacenan en una lista de diccionarios.
        - Cada oración es rodeada por etiquetas <s> y </s> para indicar su inicio y fin.
        - Las oraciones que contienen una sola palabra o están vacías se descartan.
    """
    # Leer el contenido del archivo XML
    xml_content = read_file(file_path)
    
    # Extraer el contenido entre las etiquetas <post> y </post> utilizando expresiones regulares
    post_matches = findall(r'<post>(.*?)</post>', xml_content, DOTALL)
    
    df_rows = []
    
    for post in post_matches:
        # Limpiar el texto extraído utilizando la función clean_text
        cleaned_post = clean_text(post.strip())
        
        # Dividir el texto limpio en oraciones
        sentences = [f'<s> {s.strip()} </s>' for s in split(r'\.\s*', cleaned_post) if len(s.strip().split()) > 1]
        
        # Crear un diccionario para cada oración con su texto y longitud
        df_rows.extend([{
            'text': s,
            # 'source': file_path,  # Puedes descomentar esta línea si deseas incluir la fuente del archivo
            'length': len(s.split())
        } for s in sentences])

    return df_rows
