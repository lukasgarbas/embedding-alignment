from typing import List, Tuple, Union, Dict, Any
import random

import torch

import flair.embeddings
import flair.nn
from flair.data import Sentence, Dictionary
from flair.datasets import DataLoader
from sklearn.neighbors import KNeighborsClassifier


class CEA(flair.nn.Classifier[Sentence]):
    def __init__(
        self,
        document_embeddings: flair.embeddings.DocumentEmbeddings,
        label_type: str,
        label_dictionary: Dictionary,
        train_corpus: List[Sentence],
        use_memory: bool = False,
        use_all_negatives: bool = False,
        flip_labels: bool = False,
        knn: int = 5,
        visualize_each_epoch: bool = False,
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
        :param train_corpus: training set (corpus.train) for the K-NN predictions.
        :param knn: number of neighbours for K-NN predictions
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
        self.knn_model = KNeighborsClassifier(n_neighbors=self.knn)

        # plot training set after each epoch
        self.visualize_each_epoch = visualize_each_epoch

        # memory approach: store a sentence for each class from a previous batch to find a pair for embedding alignment
        self.use_memory = use_memory
        if self.use_memory:
            self.memory = {
                label: None
                for label in self.label_dictionary.get_items()
                if label != "<unk>"
            }

        # use all possible negative pairs
        self.use_all_negatives = use_all_negatives

        # loss function: MSE between cosine similarities and 0s (pairs of different class) and 1s (pairs of same class)
        self.loss_function = torch.nn.MSELoss(reduction="sum")

        # used for multitask experiment when classifying political articles into categories
        self.flip_labels = flip_labels

        # auto-spawn on GPU if available
        self.to(flair.device)

    @property
    def label_type(self):
        return self._label_type

    @property
    def train_corpus(self):
        return self._train_corpus

    def _find_sentence_pair_in_batch(
        self,
        sentence: Sentence,
        mini_batch: List[Sentence],
        sample: str = "positive",  # 'positive' or 'negative' sample
    ) -> List[Sentence]:
        """Finds a list of all positive or negative sentence pairs in the current batch"""

        label = sentence.get_label(self._label_type).value

        # positive sentences
        positive_sentences = [
            sentence_pair
            for sentence_pair in mini_batch
            if sentence_pair.get_label(self._label_type).value == label
            and sentence_pair != sentence
        ]

        # negative sentences
        negative_sentences = [
            sentence_pair
            for sentence_pair in mini_batch
            if sentence_pair not in positive_sentences and sentence_pair != sentence
        ]

        sentence_pair: List[Sentence] = (
            positive_sentences if sample == "positive" else negative_sentences
        )
        sentence_pair = [sample for sample in sentence_pair if sample is not None]

        # use only one sample and return a single random sentence
        if not self.use_all_negatives and sentence_pair:
            sentence_pair = [random.choice(sentence_pair)]

        return sentence_pair

    def _find_sentence_pair_in_memory(
        self,
        sentence: Sentence,
        sample: str = "positive",  # 'positive' or 'negative' sample
    ) -> List[Sentence]:
        """Finds a list of all positive or negative sentence pairs in memory"""

        label = sentence.get_label(self._label_type).value

        positive_sentences = [self.memory[label]]
        negative_labels = [
            negative_label
            for negative_label in list(self.memory.keys())
            if negative_label != label
        ]

        negative_sentences = [
            self.memory[negative_label] for negative_label in negative_labels
        ]

        sentence_pair: List[Sentence] = (
            positive_sentences if sample == "positive" else negative_sentences
        )

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

        # get mini batch size
        mini_batch_size = len(sentences)

        for sentence in sentences:
            positive_samples, negative_samples = [], []

            if self.use_memory:
                positive_samples = self._find_sentence_pair_in_memory(
                    sentence, sample="positive"
                )
                negative_samples = self._find_sentence_pair_in_memory(
                    sentence, sample="negative"
                )

            # if no samples found in memory, take it from the current batch
            if not positive_samples:
                positive_samples = self._find_sentence_pair_in_batch(
                    sentence, mini_batch=sentences, sample="positive"
                )
            if not negative_samples:
                negative_samples = self._find_sentence_pair_in_batch(
                    sentence, mini_batch=sentences, sample="negative"
                )

            # add sentence pair from the same class
            for sample in positive_samples:
                first_sentences.append(sentence)
                second_sentences.append(sample)
                labels.append(0) if self.flip_labels else labels.append(1)

            # add sentence pair from a different class
            if not self.flip_labels:
                for sample in negative_samples:
                    first_sentences.append(sentence)
                    second_sentences.append(sample)
                    labels.append(1) if self.flip_labels else labels.append(0)

        if self.use_memory:
            # embed all second sentences and remove duplicates before embedding
            loader = DataLoader(list(set(second_sentences)), batch_size=mini_batch_size)
            for batch in loader:
                self.embeddings.embed(batch)

            # refresh memory after each forward pass
            for sentence in sentences:
                self.memory[sentence.get_label().value] = sentence

        first_embeddings = torch.stack(
            [sentence.get_embedding(embedding_names) for sentence in first_sentences]
        )
        second_embeddings = torch.stack(
            [sentence.get_embedding(embedding_names) for sentence in second_sentences]
        )

        # return MSE loss between sentence pair similarities and 0s and 1s
        return self._calculate_loss(
            first_embeddings, second_embeddings, torch.FloatTensor(labels)
        )

    def _calculate_loss(
        self,
        first_embeddings: List[torch.tensor],
        second_embeddings: List[torch.Tensor],
        labels: torch.FloatTensor,
    ) -> Tuple[torch.Tensor, int]:
        # put to gpu
        first_embeddings = first_embeddings.to(flair.device)
        second_embeddings = second_embeddings.to(flair.device)
        labels = labels.to(flair.device)

        # calculate cosine similarities for a full batch
        similarities = torch.nn.functional.cosine_similarity(
            first_embeddings, second_embeddings, dim=1
        )

        # MSE loss between cosine similarities and 0s (pairs of different class) and 1s (pairs of same class)
        loss = self.loss_function(similarities, labels)

        return loss, first_embeddings.shape[0]

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
        predicted_labels = self.knn_model.predict(
            torch.stack([sent.embedding for sent in sentences]).cpu()
        )
        for sentence, predicted_label in zip(sentences, predicted_labels):
            sentence.add_label(typename=label_name, value=predicted_label)

        # KNN predictions do not have loss value
        if return_loss:
            return 0

    def train(self, training: bool = True):
        super().train(training)

        if training:
            # clear embeddings from the training set after each epoch
            for sentence in self.train_corpus:
                sentence.clear_embeddings(self.embeddings.get_names())
        else:
            # embed the full training set before the evaluation (needed for knn predictions)
            # TODO: batch size is hardcoded here, think of a not to do that
            for batch in DataLoader(self.train_corpus, batch_size=8):
                self.embeddings.embed(batch)

            # prepare KNN model for faster retrieval
            embeddings = torch.stack(
                [sent.embedding for sent in self.train_corpus]
            ).cpu()

            labels = [
                sent.get_label(self.label_type).value for sent in self.train_corpus
            ]

            self.knn_model.fit(embeddings, labels)

            if self.visualize_each_epoch:
                self.visualize(self.train_corpus)

    def visualize(
        self,
        sentences: List[Sentence],
        description: str = "CEA visualization",
        mini_batch_size: int = 32,
    ):
        if not hasattr(self.knn_model, "classes_"):
            print("fitting KNN model")
            for batch in DataLoader(self.train_corpus, batch_size=mini_batch_size):
                self.embeddings.embed(batch)
            embeddings = torch.stack(
                [sent.embedding for sent in self.train_corpus]
            ).cpu()
            labels = [
                sent.get_label(self.label_type).value for sent in self.train_corpus
            ]
            self.knn_model.fit(embeddings, labels)

        self.predict(sentences)
        embeddings = torch.stack([sent.embedding for sent in sentences]).cpu()
        labels = [sent.get_label("predicted").value for sent in sentences]

        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2)
        z = tsne.fit_transform(embeddings)

        import seaborn
        import pandas

        df = pandas.DataFrame()
        df["y"], df["comp-1"], df["comp-2"] = labels, z[:, 0], z[:, 1]
        return seaborn.scatterplot(
            x="comp-1", y="comp-2", hue=df.y.tolist(), data=df
        ).set(title=description)

    def _get_state_dict(self):
        state = super()._get_state_dict()

        # add variables of DefaultClassifier
        state["document_embeddings"] = self.embeddings
        state["label_dictionary"] = self.label_dictionary
        state["label_type"] = self.label_type
        state["knn"] = self.knn
        state["use_memory"] = self.use_memory
        state["use_all_negatives"] = self.use_all_negatives
        state["flip_labels"] = self.flip_labels
        state["train_corpus"] = self.train_corpus

        return state

    @classmethod
    def _init_model_with_state_dict(cls, state: Dict[str, Any], **kwargs):
        return super()._init_model_with_state_dict(
            state,
            document_embeddings=state["document_embeddings"],
            label_dictionary=state["label_dictionary"],
            label_type=state["label_type"],
            knn=state["knn"],
            use_memory=state["use_memory"],
            use_all_negatives=state["use_all_negatives"],
            flip_labels=state["flip_labels"],
            train_corpus=state["train_corpus"],
            **kwargs,
        )
