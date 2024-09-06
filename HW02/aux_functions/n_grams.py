from pandas import Series
from collections import Counter, defaultdict
from itertools import chain



def create_ngrams(sentence: str, n: int, unique_tokens: dict = None) -> list:
    """
    Genera n-gramas a partir de una oración dada.

    Args:
        sentence (str): La oración de entrada de la cual se generarán los n-gramas.
        n (int): La longitud de los n-gramas a generar.
        unique_tokens (dict, opcional): Un diccionario que mapea ciertos tokens a un valor único,
            típicamente utilizado para reemplazar tokens poco frecuentes por un marcador como '<UNK>'.
            Por defecto es None.

    Returns:
        list: Una lista de n-gramas, donde cada n-grama se representa como una tupla de palabras.

    Notas:
        - Si se proporciona `unique_tokens`, cualquier palabra en la oración que aparezca en 
          `unique_tokens` será reemplazada según el mapeo en `unique_tokens`.
        - La función devuelve n-gramas en forma de un generador para ahorrar memoria, 
          especialmente útil al procesar grandes corpus.
    """
    words = sentence.split()  # Divide la oración en palabras.
    
    if unique_tokens:
        # Reemplaza las palabras según el diccionario unique_tokens si se proporciona.
        return (tuple([unique_tokens.get((w,), w) for w in words[i:i+n]]) 
                for i in range(len(words) - n + 1))
    
    # Genera los n-gramas sin realizar reemplazos.
    return (tuple(words[i:i+n]) for i in range(len(words) - n + 1) if len(words[i:i+n]) == n)


def create_ngram_model(n_gram: int, text_corpus: Series)-> tuple[Counter[int], defaultdict[int]]:
    """
    Crea un modelo de n-gramas a partir de un corpus de texto.

    Args:
        n_gram (int): El tamaño de los n-gramas a generar.
        text_corpus (pd.Series): Una serie de Pandas que contiene el corpus de texto, 
                                 donde cada entrada es una oración o texto.

    Returns:
        tuple: Dos elementos:
            - ngram_counts (Counter): Un contador que almacena las frecuencias de los n-gramas en el corpus.
            - final_unigram (defaultdict): Un diccionario con los (n-1)-gramas más frecuentes y su conteo, 
                                           con un marcador especial `<UNK>` para los menos frecuentes.

    Notas:
        - La función primero cuenta los (n-1)-gramas para identificar los tokens únicos que serán 
          reemplazados por `<UNK>`.
        - Luego, se cuentan los n-gramas completos utilizando los reemplazos identificados.
    """    

    # Contar las frecuencias de (n-1)-gramas en el corpus de texto.
    unigram_counts = Counter(chain.from_iterable(
        text_corpus.apply(lambda x: create_ngrams(x, n=n_gram-1, unique_tokens=None))
    ))

    final_unigram = defaultdict(int)
    unique_tokens = defaultdict(int)

    # Identificar tokens únicos y construir el diccionario final de 'unigrama'.
    for ngram, count in unigram_counts.items():
        if '' in ngram:
            continue
        if count < 2:
            unique_tokens[ngram] = ('<UNK>',) * (n_gram-1)
        else:
            final_unigram[ngram] = count

    final_unigram[('<UNK>',)*(n_gram-1)] = len(unique_tokens)

    # Contar las frecuencias de los n-gramas completos usando los reemplazos identificados.
    ngram_counts = Counter(chain.from_iterable(
        text_corpus.apply(lambda x: create_ngrams(x, n=n_gram, unique_tokens=unique_tokens))
    ))

    return ngram_counts, final_unigram


def estimate_probability(token_text: list[str], n_gram: int, 
                         final_unigram: dict, ngram_counts: dict) -> float:
    """
    Estima la probabilidad de un n-grama dado usando un modelo de n-gramas.

    Args:
        token_text (list[str]): La secuencia de tokens para la cual se desea estimar la probabilidad.
        n_gram (int): El tamaño del n-grama.
        final_unigram (dict): El diccionario que contiene las frecuencias de los (n-1)-gramas.
        ngram_counts (dict): El diccionario que contiene las frecuencias de los n-gramas.

    Raises:
        ValueError: Si el número de tokens en `token_text` no coincide con `n_gram`.

    Returns:
        float: La probabilidad estimada del n-grama dado.

    Notas:
        - La función utiliza suavizado de Laplace para evitar probabilidades de cero.
        - Si el prefijo (n-1)-grama no se encuentra en `final_unigram`, se reemplaza con (`<UNK>`,)* (n_gram-1).
    """    
    
    if len(token_text) != n_gram:
        raise ValueError(f'El texto de entrada debe tener {n_gram} tokens')
    
    # Reemplaza el prefijo con `<UNK>` si no se encuentra en final_unigram.
    if not final_unigram.get(tuple(token_text[:n_gram-1])):
        token_text = list(token_text)
        token_text[:n_gram] = ('<UNK>',) * (n_gram-1)
        token_text = tuple(token_text)

    # Suavizado de Laplace
    # Probabilidad del (n-1)-grama
    p_wi = final_unigram.get(tuple(token_text[:n_gram-1]), 0) + len(final_unigram)
    
    # Probabilidad del n-grama completo dado su (n-1)-grama
    p_w_wi = (ngram_counts.get(tuple(token_text), 1)) / p_wi
        
    return p_w_wi