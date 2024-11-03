# Regulatory Information Retrieval And Answer Generation Shared Task - Codabench competition #3527

This repo contains our solution to the competition [Regulatory Information Retrieval and Answer Generation](https://www.codabench.org/competitions/3527) addressing the following subtask.

_Passage Retrieval_. Given a regulatory question, we developed an information retrieval (IR) system that returns the most relevant passages from ADGM regulations and guidance documents. The challenge is to design a system that can retrieve these passages given the complex language and expertise required to understand the regulatory documents.

The repository contains three notebooks detailed below.

- `data_processing.ipynb`: This notebook processes the competition dataset and outputs training, testing, and validation datasets ready for use.

- `model.ipynb`: This notebook fine-tunes a text embedding model which is later used for the information retrieval task.

- `ObligQA.ipynb`: This notebook solves the information retrieval task and demonstrates different approaches, out of which the hybrid approach which uses both lexical and semantic techniques performs best.


1. **Data Preprocessing**: The provided datasets were cleaned and preprocessed to ensure consistency and quality. This included tokenization, normalization, and removal of irrelevant information.

2. **Model Selection**: We selected the `BAAI/bge-small-en-v1.5` model from Hugging Face, which is a fine-tuned version of the BERT model widely used in natural language processing.

3. **Training**: The model was further fine-tuned on the competition dataset to improve its performance.

4. **Evaluation**: The model was evaluated using common information retrieval metrics such as `Recall@10` and `MAP@10`.

5. **Information Retrieval**: The final model was used to generate answers for the test set, which were then submitted to the competition platform achieving a recall rate of 0.833 surpassing by more than 0.05 points the baseline.

## Results

The model achieved good results in the competition rankings. Detailed results and analysis can be found in the report `NLP-Informe`.

## Usage

To reproduce the results, follow these steps:

- Use python 3.10

```sh
python --version
```

- Create virtual environment

```sh
python -m venv env
```

- Activate virtual environment

```sh
.\env\Scripts\activate
```

- Install the required dependencies

 ```bash
pip install -r requirements.txt
```

## References

- [Codabench Competition 3527](https://www.codabench.org/competitions/3527)
- [Hugging Face Model](https://huggingface.co/raul-delarosa99/bge-small-en-v1.5-RIRAG_ObliQA)