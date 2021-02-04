# -*- coding:utf8 -*-

import torch
import torch.nn as nn
import torch.optim
import numpy as np

def word2index(word, vocab):
    """
    Convert an word token to an dictionary index
    """
    if word in vocab:
        value = vocab[word][0]
    else:
        value = -1
    return value


def index2word(index, vocab):
    """
    Convert an word index to a word token
    """
    for w, v in vocab.items():
        if v[0] == index:
            return w
    return 0


class Model(object):
    def __init__(self, args, vocab, pos_data, neg_data):
        """The Text Classification model """
        self.embeddings_dict = {}
        self.algo = args.algo
        if self.algo == "GLOVE":
            print("Now we use the glove embedding")
            self.load_glove(args.emb_file)
        self.vocab = vocab
        self.pos_sentences = pos_data
        self.neg_sentences = neg_data
        self.lr = args.lr
        self.embed_size = args.embed_size
        self.hidden_size =args.hidden_size
        self.dataset = []
        self.labels = []
        self.sentences = []

        self.train_data = []
        self.train_label = []

        self.valid_data = []
        self.valid_label = []

        self.test_data = []
        self.test_label = []

        if self.algo == "GLOVE":
            self.model = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2),
            nn.LogSoftmax(),)
        else:
            self.model = nn.Sequential(
                nn.Linear(len(vocab), self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 2),
                nn.LogSoftmax(), )

    def load_dataset(self):
        """
        Load the training and testing dataset
        """
        for sentence in self.pos_sentences:
            new_sentence = []
            for l in sentence:
                if l in self.vocab:
                    if self.algo == "GLOVE":
                        new_sentence.append(l)
                    else:
                        new_sentence.append(word2index(l, self.vocab))
            self.dataset.append(self.sentence2vec(new_sentence, self.vocab))
            self.labels.append(0)
            self.sentences.append(sentence)

        for sentence in self.neg_sentences:
            new_sentence = []
            for l in sentence:
                if l in self.vocab:
                    if self.algo == "GLOVE":
                        new_sentence.append(l)
                    else:
                        new_sentence.append(word2index(l, self.vocab))
            self.dataset.append(self.sentence2vec(new_sentence, self.vocab))
            self.labels.append(1)
            self.sentences.append(sentence)

        indices = np.random.permutation(len(self.dataset))

        self.dataset = [self.dataset[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        self.sentences = [self.sentences[i] for i in indices]

        # split dataset
        test_size = len(self.dataset) // 10
        self.train_data = self.dataset[2 * test_size:]
        self.train_label = self.labels[2 * test_size:]

        self.valid_data = self.dataset[: test_size]
        self.valid_label = self.labels[: test_size]

        self.test_data = self.dataset[test_size: 2 * test_size]
        self.test_label = self.labels[test_size: 2 * test_size]

    def rightness(self, predictions, labels):
        """ 
        Prediction of the error rate
        """
        pred = torch.max(predictions.data, 1)[1]  #
        rights = pred.eq(labels.data.view_as(pred)).sum()  #
        return rights, len(labels)  #

    def sentence2vec(self, sentence, dictionary):
        """ 
        Convert sentence text to vector representation 
        """
        if self.algo == "GLOVE":
            vector = np.zeros(self.embed_size)
            for word in sentence:
                if word in self.embeddings_dict:
                    vector += self.embeddings_dict[word]
            return 1.0 * vector / len(sentence)
        else:
            vector = np.zeros(len(dictionary))
            for l in sentence:
                vector[l] += 1
            return 1.0 * vector / len(sentence)

    def load_glove(self, path):
        """
        Load Glove embeddings dictionary
        """
        with open(path, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector
        return 0

    def training(self):
        """
        The whole training and testing process.
        """
        losses = []
        loss_function = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(50):
            print(epoch)
            for i, data in enumerate(zip(self.train_data, self.train_label)):
                x, y = data
                x = torch.tensor(x, requires_grad=True, dtype=torch.float).view(1, -1)
                y = torch.tensor(np.array([y]), dtype=torch.long)
                optimizer.zero_grad()
                # predict
                predict = self.model(x)
                # calculate loss
                loss = loss_function(predict, y)
                losses.append(loss.data.numpy())
                loss.backward()
                optimizer.step()
                # test every 1000 epoch
                if i % 1000 == 0:
                    val_losses = []
                    rights = []
                    for j, val in enumerate(zip(self.valid_data, self.valid_label)):
                        x, y = val
                        x = torch.tensor(x, requires_grad=True, dtype=torch.float).view(1, -1)
                        y = torch.tensor(np.array([y]), dtype=torch.long)
                        predict = self.model(x)
                        right = self.rightness(predict, y)
                        rights.append(right)
                        loss = loss_function(predict, y)
                        val_losses.append(loss.data.numpy())

                    right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
                    print('At the {} epoch，Training loss：{:.2f}, Testing loss：{:.2f}, Testing Acc: {:.2f}'.format(epoch, np.mean(losses),
                                                                                np.mean(val_losses), right_ratio))
        print("Training End")




