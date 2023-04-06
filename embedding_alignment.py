from flair.datasets import CSVClassificationCorpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import EmbeddingAlignmentClassifier
from flair.trainers import ModelTrainer

# specify gpu device
import flair, torch
flair.device = torch.device('cuda:0')

# prepare multiple runs
model_handle = "google/electra-small-discriminator"
parameters = [
    (5e-5, 32, 10),
    (5e-5, 32, 10),
    (5e-5, 32, 10),
]

# 1. load custom corpus from CSV
data_directory = "data/fixedsplits/bias"
label_map = {0: "label", 2: "text"}  # label at column 0 is bias {left, right, center}, column 2 is for an article

corpus = CSVClassificationCorpus(data_folder=data_directory,
                                 column_name_map=label_map,
                                 label_type="label",
                                 skip_header=True,
                                 train_file='train.csv',
                                 dev_file='dev.csv',
                                 test_file='test.csv',
                                 in_memory=True,  # keep corpus + embeddings in memory for KNN evaluation
                                 )

print(corpus)

# 2. create the label dictionary
label_type = 'label'
label_dict = corpus.make_label_dictionary(label_type=label_type)

for parameter in parameters:

    learning_rate, batch_size, epochs = parameter

    # 3. initialize transformer document embeddings
    document_embeddings = TransformerDocumentEmbeddings(model_handle, fine_tune=True)

    # 4. create embedding alignment model
    classifier = EmbeddingAlignmentClassifier(document_embeddings,
                                              train_corpus=corpus.train,
                                              label_type=label_type,
                                              label_dictionary=label_dict,
                                              use_memory=False,
                                              knn=5)

    # 5. initialize trainer
    trainer = ModelTrainer(classifier, corpus)

    # 6. run training with fine-tuning
    trainer.fine_tune('resources/taggers/transformer-alignment',
                      learning_rate=learning_rate,
                      mini_batch_size=batch_size,
                      max_epochs=epochs,
                      )