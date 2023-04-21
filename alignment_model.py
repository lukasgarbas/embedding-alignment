import logging
from typing import List, Tuple, Union
from collections import Counter
import random

import torch

import flair.embeddings
import flair.nn
from flair.data import Sentence, Dictionary
from flair.datasets import DataLoader


class CEA(flair.nn.Classifier[Sentence]):
    def __init__(
        self,
        document_embeddings: flair.embeddings.DocumentEmbeddings,
        label_type: str,
        label_dictionary: Dictionary,
        train_corpus: List[Sentence],
        use_memory: bool = False,
        use_all_negatives: bool = False,
        knn: int = 5,
        **classifierargs,
    ):
        """
        Class-driven Embedding Alignment Model (CEA)
        The main idea is to learn embeddings that maximize the similarity between two
        textual documents if they share the same class label, and minimize it if they do not.
        This model does not have any learnable parameters, thus it supports only Transformer Embeddings
        because they can be fine-tuned. The model embeds text documents, creates text pairs, and aligns
        two documents closer to each other (by optimizing cosine similarity to be equal to 1) if they belong
        to the same class and moves two documents apart (similarity to be equal to 0) if they don't.
        Prediction phase uses learned embeddings and K-Nearest Neighbors to classify documents.

        :param document_embeddings: Embedding used to encode sentence (transformer document embeddings for now)
        :param label_type: Name of the gold labels to use.
        :param train_corpus: The model uses corpus.train training set for the KNN algorithm.
        :param knn: number of neighbours for KNN predictions
        :param use_memory: store a sentence from previous batch
        :param use_all_negatives: use all available negative samples
        """

        super(CEA, self).__init__(**classifierargs)

        # only document embeddings so far
        self.embeddings: flair.embeddings.DocumentEmbeddings = document_embeddings

        # corpus is required to make KNN predictions
        self._train_corpus: List[Sentence] = train_corpus

        self._label_type: str = label_type

        self.label_dictionary: Dictionary = label_dictionary

        # number of neighbours for K-NN predictions
        self.knn = knn

        # memory approach: store a sentence for each class from a previous batch to find a pair for embedding alignment
        self.use_memory = use_memory
        if self.use_memory:
            self.memory = {label: None for label in self.label_dictionary.get_items() if label != "<unk>"}

        # use all possible negative pairs
        self.use_all_negatives = use_all_negatives

        # loss function: MSE between cosine similarities and 0s (pairs of different class) and 1s (pairs of same class)
        self.loss_function = torch.nn.MSELoss(reduction="sum")

        # auto-spawn on GPU if available
        self.to(flair.device)

    @property
    def label_type(self):
        return self._label_type

    @property
    def train_corpus(self):
        return self._train_corpus

    def _find_sentence_pair_in_batch(self,
                                     sentence: Sentence,
                                     mini_batch: List[Sentence],
                                     sample: str = "positive",  # 'positive' or 'negative' sample
                                     ) -> List[Sentence]:
        """Finds a list of all positive or negative sentence pairs in the current batch"""

        label = sentence.get_label(self._label_type).value

        # positive sentences
        positive_sentences = [sentence_pair for sentence_pair in mini_batch
                              if sentence_pair.get_label(self._label_type).value == label
                              and sentence_pair != sentence]

        # negative sentences
        negative_sentences = [sentence_pair for sentence_pair in mini_batch
                              if sentence_pair not in positive_sentences
                              and sentence_pair != sentence]

        sentence_pair: List[Sentence] = positive_sentences if sample == "positive" else negative_sentences
        sentence_pair = [sample for sample in sentence_pair if sample is not None]

        # use only one sample and return a single random sentence
        if not self.use_all_negatives and sentence_pair:
            sentence_pair = [random.choice(sentence_pair)]

        return sentence_pair

    def _find_sentence_pair_in_memory(self,
                                      sentence: Sentence,
                                      sample: str = "positive",  # 'positive' or 'negative' sample
                                      ) -> List[Sentence]:
        """Finds a list of all positive or negative sentence pairs in memory"""

        label = sentence.get_label(self._label_type).value

        positive_sentences = [self.memory[label]]
        negative_labels = [negative_label for negative_label in list(self.memory.keys()) if negative_label != label]
        negative_sentences = [self.memory[negative_label] for negative_label in negative_labels]

        sentence_pair: List[Sentence] = positive_sentences if sample == "positive" else negative_sentences
        sentence_pair = [sample for sample in sentence_pair if sample is not None]

        # use only one sample and return a single random sentence
        if not self.use_all_negatives and sentence_pair:
            sentence_pair = [random.choice(sentence_pair)]

        return sentence_pair

    def forward_loss(self, sentences: List[Sentence]) -> Tuple[torch.Tensor, int]:

        # embed all sentences inside a batch
        self.embeddings.embed(sentences)
        embedding_names = self.embeddings.get_names()

        first_sentences: List[Sentence] = []
        second_sentences: List[Sentence] = []
        labels: List[torch.tensor] = []

        for sentence in sentences:
            positive_samples, negative_samples = [], []

            if self.use_memory:
                positive_samples = self._find_sentence_pair_in_memory(sentence, sample="positive")
                negative_samples = self._find_sentence_pair_in_memory(sentence, sample="negative")

            # if no samples found in memory, take it from the current batch
            # this is necessary for the first forward pass even if using the memory
            if not positive_samples:
                positive_samples = self._find_sentence_pair_in_batch(sentence, mini_batch=sentences, sample="positive")
            if not negative_samples:
                negative_samples = self._find_sentence_pair_in_batch(sentence, mini_batch=sentences, sample="negative")

            # add sentence pair from the same class and label 1
            for sample in positive_samples:
                first_sentences.append(sentence)
                second_sentences.append(sample)
                labels.append(1)

            # add sentence pair from a different class and label 0
            for sample in negative_samples:
                first_sentences.append(sentence)
                second_sentences.append(sample)
                labels.append(0)

        if self.use_memory:
            # embed all second sentences and remove duplicates before embedding
            self.embeddings.embed(list(set(second_sentences)))

            # refresh memory after each forward pass
            for sentence in sentences:
                self.memory[sentence.get_label().value] = sentence

        first_embeddings = torch.stack([sentence.get_embedding(embedding_names) for sentence in first_sentences])
        second_embeddings = torch.stack([sentence.get_embedding(embedding_names) for sentence in second_sentences])

        # return MSE loss between sentence pair similarities and 0s and 1s
        return self._calculate_loss(first_embeddings, second_embeddings, torch.FloatTensor(labels))

    def _calculate_loss(self,
                        first_embeddings: List[torch.tensor],
                        second_embeddings: List[torch.Tensor],
                        labels: torch.FloatTensor,
                        ) -> Tuple[torch.Tensor, int]:

        # put to gpu
        first_embeddings = first_embeddings.to(flair.device)
        second_embeddings = second_embeddings.to(flair.device)
        labels = labels.to(flair.device)

        # calculate cosine similarities for a full batch
        similarities = torch.nn.functional.cosine_similarity(first_embeddings, second_embeddings, dim=1)

        # MSE loss between cosine similarities and 0s (pairs of different class) and 1s (pairs of same class)
        loss = self.loss_function(similarities, labels)

        return loss, first_embeddings.shape[0]

    def _knn(self, sentence: Sentence) -> str:
        """KNN classification for a single sentence"""

        # get embedding for a given sentence
        embedding_names = self.embeddings.get_names()
        sentence_embedding = sentence.get_embedding(embedding_names)

        # gather embeddings for all training instances
        train_set_embeddings = [training_sample.get_embedding(embedding_names)
                                for training_sample in self.train_corpus]
        train_set_embeddings = torch.stack(train_set_embeddings)

        # calculate cos similarity between given sentence and all training instances
        similarities = torch.nn.functional.cosine_similarity(sentence_embedding, train_set_embeddings, dim=1)

        # sort by top k (nearest neighbours) and get their labels
        _, top_k_idx = similarities.topk(k=self.knn)
        closest_labels = [self.train_corpus[sentence_id].get_label(self._label_type).value
                          for sentence_id in top_k_idx]

        # majority vote
        predicted_label = Counter(closest_labels).most_common(1)[0][0]

        return predicted_label

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence],
        mini_batch_size: int = 32,
        label_name: str = "predicted",
        return_loss: bool = False,
        **kwargs,
    ):
        """Predictions use K-Nearest Neighbor thus needs embeddings of the training set"""

        if isinstance(sentences, Sentence):
            sentences = [sentences]

        # filter empty sentences
        sentences = [sentence for sentence in sentences if len(sentence) > 0]
        if len(sentences) == 0:
            return sentences

        # embed sentences to be predicted
        loader = DataLoader(sentences, batch_size=mini_batch_size)
        for batch in loader:
            self.embeddings.embed(batch)

        # perform KNN to assign each new sentence a class
        for sentence in sentences:
            predicted_label = self._knn(sentence)
            sentence.add_label(typename=label_name, value=predicted_label)

        # KNN predictions do not have loss value
        if return_loss:
            return 0

    def train(self, training: bool = True):
        if training:
            # clear embeddings from the training set after each epoch
            for sentence in self.train_corpus:
                sentence.clear_embeddings(self.embeddings.get_names())
        else:
            # embed the full training set before the evaluation (needed for knn predictions)
            loader = DataLoader(self.train_corpus, batch_size=32)
            for batch in loader:
                self.embeddings.embed(batch)

        return super().train(training)
