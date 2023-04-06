# CAE - Class-driven Embedding Alignment

Class-driven Embedding Alignment Model (CEA). Learn embeddings that maximize the similarity between two textual documents if they share the same class label 👉👈, and minimize if they don't 👈👉.

```
git clone https://github.com/flairNLP/flair.git
cd flair

# switch to cea branch
git checkout embedding-alignment

python3 -m venv flair-env
pip install -r requirements.txt
```

# Training on TREC dataset

```python3
from flair.datasets import TREC_6
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import EmbeddingAlignmentClassifier
from flair.trainers import ModelTrainer

# specify gpu device
import flair, torch
flair.device = torch.device('cuda:0')

# load dataset
corpus = TREC_6()

# specify label type
label_type = 'question_class'  # just 'class' for TREC_50

# create the label dictionary
label_dict = corpus.make_label_dictionary(label_type=label_type)

print(corpus)

# initialize transformer document embeddings
document_embeddings = TransformerDocumentEmbeddings('google/electra-small-discriminator', fine_tune=True)

# create embedding alignment model
classifier = EmbeddingAlignmentClassifier(document_embeddings,
                                          train_corpus=corpus.train,
                                          label_type=label_type,
                                          label_dictionary=label_dict,
                                          knn=5,  # k for knn predictions
                                          )

# initialize trainer
trainer = ModelTrainer(classifier, corpus)

# run training with fine-tuning
trainer.fine_tune('resources/taggers/small-transformer-alignment',
                  learning_rate=5e-5,
                  mini_batch_size=32,
                  max_epochs=10,
                  )
```

# Results

- parameters: learning rate 5e-5, batch size 32, epochs 10

| model           | details             | Trec 6   | Trec 50   |
|-----------------|---------------------|----------|-----------|
| electra-small   | fine-tuning         | 96.2     | 89.8      |
| electra-small   | CEA                 | 96.2     | 84.0      |
| electra-small   | CEA + memory        | 96.0     | 81.4      |
| electra-small   | CEA + mixed memory  | 96.4     | 84.8      |
| --------------- | ------------------- | -------- | --------- |
| bert-base       | fine-tuning         | 97.4     | 92.6      |
| bert-base       | CEA                 | 97.2     | 90.6      |
| bert-base       | CEA + memory        | 96.4     | 90.6      |
| bert-base       | CEA + mixed memory  | 97.2     | 90.8      |

Takeaway:
- CEA can barely recreate scores of fine-tuning (i.e. cea scores slightly lower or the same as fine-tuning).
- Hard with a lot of classes (TREC_50); this can be due to my implementation where sampling for sentence pairs is done inside a batch.


# Political Bias Dataset

- Transformers to consider:
  - electra-small (just because of the speed and gpu memory)
  - bert-base (classic)
  - longformer (probably the best choice for long documents). Longformer has a [smaller version (1096 tokens)](kiddothe2b/longformer-mini-1024) and [medium (4096 tokens)](allenai/longformer-base-4096) 👈 This one would be to go, just barely fits anywhere (setup: gruenau9 + batch size 4 or 8)

### About the dataset

- Labels: skews right, skews left, center / more reliable
- Corpus: 8640 train + 960 dev + 1200 test sentences
- train set size: 8640 (2880 instances for each class)
- dev set size: 960 (320 for each class)
- test set size: 1200 (400 for each class)
- Publishers in train and dev sets: Fox News, The Nation, AP, Washington Times, The Intercept, Reuters
- Publishers in test set: The Washington, Free Beacon, The New Yorker, CNBC

# Results and TODOs

| model           | details              | dev F1   | test F1   |
|-----------------|----------------------|----------|-----------|
| electra-small   | fine-tuning          | 94.34    | 67.6      |
| electra-small   | CEA                  | 95.0     | 67.0      |
| electra-small   | CEA + memory         | -        | -         |
| electra-small   | CEA + mixed memory   | -        | -         |
| electra-small   | CEA + multitask      | -        | -         |
| --------------- | -------------------- | -------- | --------- |
| longformer-base | fine-tuning          | 98.43    | 71.04     |
| longformer-base | CEA + multitask      | -        | -         |


TODO: run multitask_alignment.py. Find out why CSVClassificationCorpus(in_memory=True) + this bias dataset + current master gives error.

- preliminary: multitask electra-small (just 2 epochs). Current walkaround when loading bias dataset `CSVClassificationCorpus(in_memory=True, tokenizer=SpaceTokenizer())`

| model          | details                         | bias dev | bias test | publisher dev | publisher test |
|----------------|---------------------------------|----------|-----------|---------------|----------------|
| electra-small  | cae + multitask (just 2 epochs) | 92.5     | 65.0      | 77.8          | 0.0            |

