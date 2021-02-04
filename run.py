# -*- coding:utf8 -*-

import argparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from classifier import Model
from collections import Counter

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def preprocess(file, label_file, is_filter=True):
    """
    Prepare the data and the vocab for the models.
    For expediency, the vocabulary will be all the words
    in the dataset (not split into training/test), so
    the assignment can avoid the OOV problem.
    """
    all_words = []
    pos_sentences = []
    neg_sentences = []
    with open(file, 'r') as fr:
        with open(label_file, 'r') as lf:
            for idx, line in enumerate(fr):
                label = lf.readline().strip()
                words = word_tokenize(line)
                if is_filter:
                    words = [w for w in words if not w in stop_words]
                    words = [word.lower() for word in words if word.isalpha()]
                if len(words) > 0:
                    all_words += words
                    if "positive" == label:
                        pos_sentences.append(words)
                    else:
                        neg_sentences.append(words)
    print('{0} contain {1} lines, {2} words.'.format(file, idx + 1, len(all_words)))
    # Built vocab : {w:[id, frequent]}
    vocab = {}
    cnt = Counter(all_words)
    for word, freq in cnt.items():
        # Note that we do not use the words whose frequency is less than 2
        if freq > 2:
            vocab[word] = [len(vocab), freq]
    print('The size of dictionary is：{}'.format(len(vocab)))
    return pos_sentences, neg_sentences, vocab


def modeling(args, pos_sentences, neg_sentences, vocab):
    """
    Init the model and start to train the model
    """
    model = Model(args, vocab, pos_sentences, neg_sentences)
    model.load_dataset()
    model.training()
    return 0


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Text Classification')
    parser.add_argument('--train', action='store_true',
                        help='if use the whole dataset')
    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BOW', 'GLOVE'], default='GLOVE',
                                help='choose the input word vector algorithm to use')
    model_settings.add_argument('--lr', type=float, default=0.001,
                                help='learning rate')
    model_settings.add_argument('--hidden_size', type=int, default=5,
                                help='the hidden size of the classifier')
    model_settings.add_argument('--embed_size', type=int, default=50,
                                help='size of the glove embeddings')
    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--emb_file', default='./data/embedding.txt',
                               help='Path of pre-trained input data')
    path_settings.add_argument('--review_file', default='./data/reviews_500.txt',
                               help='Path of the input reviews text')
    path_settings.add_argument('--label_file', default='./data/labels_500.txt',
                               help='Path of the input reviews label')
    return parser.parse_args()


def run():
    """
    Prepares and runs the whole model. 
    """
    args = parse_args()
    pos_data, neg_data, vocab = preprocess(args.review_file, args.label_file, True)
    modeling(args, pos_data, neg_data, vocab)

if __name__ == '__main__':
    run()
