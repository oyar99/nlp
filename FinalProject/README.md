# Regulatory Information Retrieval And Answer Generation Shared Task - Codabench competition #3527

This repo contains our solution to the competition [Regulatory Information Retrieval and Answer Generation](https://www.codabench.org/competitions/3527) addressing the following subtask.

_Passage Retrieval_. Given a regulatory question, we developed an information retrieval (IR) system that returns the most relevant passages from ADGM regulations and guidance documents. The challenge is to design a system that can retrieve these passages given the complex language and expertise required to understand the regulatory documents.

1. **Data Preprocessing**: The provided datasets were cleaned and preprocessed to ensure consistency and quality. This included tokenization, normalization, and removal of irrelevant information.

2. **Model Selection**: We selected the `bge-small-en-v1.5-RIRAG_ObliQA` model from Hugging Face, which is a fine-tuned version of the BERT model specifically designed for question answering tasks.

3. **Training**: The model was further fine-tuned on the competition dataset to improve its performance. We used techniques such as learning rate scheduling, early stopping, and data augmentation to enhance the training process.

4. **Evaluation**: The model was evaluated using the competition's metrics, and hyperparameters were tuned based on the validation set performance.

5. **Inference**: The final model was used to generate answers for the test set, which were then submitted to the competition platform.

## Results

The model achieved a high accuracy on the validation set and performed well in the competition rankings. Detailed results and analysis can be found in the `results` folder.

## Usage

To reproduce the results, follow these steps:

- Use python 3.9

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

## Conclusion

This solution demonstrates the effectiveness of fine-tuning pre-trained models for specific tasks. By leveraging advanced NLP techniques and thorough evaluation, we were able to achieve competitive results in the Codabench Competition 3527.

## References

- [Codabench Competition 3527](https://www.codabench.org/competitions/3527)
- [Hugging Face Model](https://huggingface.co/raul-delarosa99/bge-small-en-v1.5-RIRAG_ObliQA)