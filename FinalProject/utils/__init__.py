from datasets import Dataset
from pandas import read_pickle


CORPUS = read_pickle('./data/corpus.pkl') # Our corpus (cid => document)


def create_data_for_evaluator(dataset:Dataset) -> dict:
    """_summary_

    Args:
        dataset (Dataset): _description_

    Returns:
        dict: _description_
    """    

    queries = dict(
        zip(dataset['anchor_id'], 
            dataset['anchor'])
    )  # Our queries (qid => question)

    # Create a mapping of relevant document (1 in our case) for each query
    relevant_docs = {qid:[] for qid in dataset['anchor_id']}  # Query ID to relevant documents (qid => set([relevant_cids])
    for qid, cid  in zip(dataset['anchor_id'], dataset['positive_id']):
        relevant_docs[qid].append(cid)
        
    return dict(corpus=CORPUS, queries=queries, relevant_docs=relevant_docs)
