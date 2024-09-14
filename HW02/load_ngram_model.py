#!/usr/bin/env python
# coding: utf-8

import json

def load_ngram_model(model_name: str) -> tuple:
    """
    Carga un modelo de n-gramas desde un archivo JSON y lo devuelve junto con el conteo de unigramas.

    Args:
        model_name (str): El nombre del archivo del modelo de n-gramas que se desea cargar (sin extensión).

    Returns:
        tuple: Un tuple que contiene:
            - ngram_counts (Counter): Un diccionario que almacena los n-gramas y sus frecuencias.
            - final_unigram (defaultdict): Un diccionario con las frecuencias de los unigramas o (n-1)-gramas más frecuentes.
    """
    try:
        with open(f'./data/ngram_models/{model_name}.json', 'r') as f:
            model_data = json.load(f)
        print(f"Modelo {model_name} cargado exitosamente.")
        
        # Se espera que los modelos estén en el formato {ngram_counts: {...}, final_unigram: {...}}
        ngram_counts = model_data.get('ngram_counts', {})
        final_unigram = model_data.get('final_unigram', {})
        
        return ngram_counts, final_unigram
    
    except FileNotFoundError:
        print(f"Error: El archivo {model_name}.json no fue encontrado.")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: No se pudo decodificar el archivo {model_name}.json. Verifica el formato del archivo.")
        return None, None
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return None, None
