from flair.datasets import CSVClassificationCorpus
from flair.embeddings import TransformerDocumentEmbeddings
from alignment_model import CEA
from flair.models import MultitaskModel
from flair.nn.multitask import make_multitask_model_and_corpus
from flair.trainers import ModelTrainer

# specify gpu device
import flair, torch
flair.device = torch.device('cuda:0')

# prepare multiple runs
model_handle = "google/electra-small-discriminator"
parameters = [
    (5e-5, 32, 20)
]

# initialize transformer document embeddings
shared_embeddings = TransformerDocumentEmbeddings(model_handle, fine_tune=True)

# 1. Create alignment model for political bias
data_directory = "data/fixedsplits/bias"
bias_label_type = "label"
bias_label_map = {0: "label", 2: "text"}  # label at column 0 is bias {left, right, center}, column 2 is for article

bias_corpus = CSVClassificationCorpus(data_folder=data_directory,
                                      column_name_map=bias_label_map,
                                      label_type=bias_label_type,
                                      skip_header=True,
                                      train_file='train.csv',
                                      dev_file='dev.csv',
                                      test_file='test.csv',
                                      in_memory=True)

print(f"Political Bias Corpus: {bias_corpus}")
bias_label_dict = bias_corpus.make_label_dictionary(label_type=bias_label_type)

bias_classifier = CEA(shared_embeddings,
                      train_corpus=bias_corpus.train,
                      label_type=bias_label_type,
                      label_dictionary=bias_label_dict,
                      use_all_negatives=True,
                      knn=5)

# 2. Create alignment model for publisher classification
data_directory = "data/fixedsplits/bias"
publisher_label_type = "label"
publisher_label_map = {1: "label", 2: "text"}  # label at column 1 is publisher, column 2 is for article

publisher_corpus = CSVClassificationCorpus(data_folder=data_directory,
                                           column_name_map=publisher_label_map,
                                           label_type=publisher_label_type,
                                           skip_header=True,
                                           train_file='train.csv',
                                           dev_file='dev.csv',
                                           test_file='test.csv',
                                           in_memory=True)

print(f"Publisher Corpus: {publisher_corpus}")
publisher_label_dict = publisher_corpus.make_label_dictionary(label_type=publisher_label_type)

publisher_classifier = CEA(shared_embeddings,
                           train_corpus=publisher_corpus.train,
                           label_type=publisher_label_type,
                           label_dictionary=publisher_label_dict,
                           use_all_negatives=True,
                           flip_labels=True,
                           knn=5)

for parameter in parameters:
    learning_rate, batch_size, epochs = parameter

    # 3. create multitask model
    multitask_model, multicorpus = make_multitask_model_and_corpus(
        [
            (bias_classifier, bias_corpus),
            (publisher_classifier, publisher_corpus),
        ]
    )

    # 4. initialize trainer
    trainer = ModelTrainer(multitask_model, multicorpus)

    # 5. run training with fine-tuning
    trainer.fine_tune('resources/taggers/multitask-transformer-alignment',
                      learning_rate=learning_rate,
                      mini_batch_size=batch_size,
                      max_epochs=epochs,
                      )
