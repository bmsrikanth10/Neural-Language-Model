# -*- coding:utf8 -*-

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

CONTEXT_SIZE = 2
EMBEDDING_DIM = 50

BATCH_SIZE = 16
BATCH_SIZE = 512
trigrams = []
vocab = {}

def genbatch(batch):
    tgt = [word2index(trigram[1], vocab) for trigram in batch]
    ctxt1 = [word2index(trigram[0][0], vocab) for trigram in batch]
    ctxt2 = [word2index(trigram[0][1], vocab) for trigram in batch]
    target = torch.tensor(tgt, dtype=torch.long)
    context = torch.tensor([ctxt1, ctxt2], dtype=torch.long)
    context = context.t() #transpose
    return context, target

def word2index(word, vocab):
    """
    Convert a word token to a dictionary index
    """
    if word in vocab:
        value = vocab[word]
    else:
        value = -1
    return value

def index2word(index, vocab):
    """
    Convert a word index to a word token
    """
    for w, v in vocab.items():
        if v == index:
            return w
    return 0

def preprocess(file, is_filter=True):
    """
    Prepare the data and the vocab for the models.
    For expediency, the vocabulary will be all the words
    in the dataset (not split into training/test), so
    the assignment can avoid the OOV problem.
    """
    with open(file, 'r') as fr:
        for idx, line in enumerate(fr):
            words = word_tokenize(line)
            if is_filter:
                words = [w for w in words if not w in stop_words]
                words = [word.lower() for word in words if word.isalpha()]
                for word in words:
                    if word not in vocab:
                        vocab[word] = len(vocab)
            if len(words) > 0:
                for i in range(len(words) - 2):
                    trigrams.append(([words[i], words[i + 1]], words[i + 2]))
    
    diff =  BATCH_SIZE - (len(trigrams) % BATCH_SIZE)
    for i in range(diff):
        trigrams.append(trigrams[i])
        
    
    print('{0} contain {1} lines'.format(file, idx + 1))
    print('The size of dictionary is：{}'.format(len(vocab)))
    print('The size of trigrams is：{}'.format(len(trigrams)))
    return 0


class NgramLM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NgramLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 256) # context size = 2, embedding_dim = 1000
        self.linear2 = nn.Linear(256, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((BATCH_SIZE, -1))
        # TODO
        hidden = self.linear1(embeds)
        out = self.linear2(hidden)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
    
#    def output(model, file1, file2):
#        """
#		Write the embedding file to the disk
#		"""
#        with open(file1, 'w') as fw1:
#            weights = model.embeddings.weight
#            print(weights.size())
#            for word, id in vocab.items():
#                # TODO
#                ostr = weights[id].data.numpy()
#                fw1.write(word)
#                fw1.write(" ")
#                fw1.write(' '.join(map(str, ostr)))
#                fw1.write("\n")
#        with open(file2, 'w') as fw2:
#            weights_2 = torch.tensor(np.random.rand(len(vocab), 50))
#            for word,id in vocab.items():
#                # TODO
#                ostr = weights_2[id].data.numpy()
#                fw2.write(word)
#                fw2.write(" ")
#                fw2.write(' '.join(map(str, ostr)))
#                fw2.write("\n")                
def output(model, file1, file2):
    with open(file1, 'w') as fw1:
        for word, id in vocab.items():
            # TODO
            ostr = []
            weights = model.embeddings.weight[id].data.numpy()
            ostr.append(word)
            #ostr.extend(' ')
            ostr.extend(weights)
#            fw1.write(ostr)
            fw1.write(' '.join(map(str, ostr)))
            fw1.write("\n")
        print('line140')
    with open(file2, 'w') as fw2:
        for word,id in vocab.items():
            ostr = []
            # TODO
            ostr.append(word)
            #ostr.extend(' ')
            ostr.extend(numpy.random.rand(50))
#            fw2.write(ostr)
            fw2.write(' '.join(map(str, ostr)))
            fw2.write("\n")
        print('line151')


def training():
    """
    Train the NLM
    """
    #preprocess('./data/reviews_100.txt')
    preprocess('./data/reviews_500.txt')
    #preprocess('./data/reviews.txt')
    #for extra credit, please uncomment the above line
    
    losses = []
    model = NgramLM(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    # TODO
    someOptimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    someFunction = torch.nn.NLLLoss()
    #someFunction = torch.nn.KLDivLoss()
    optimizer = someOptimizer
    loss_function = someFunction
    samp = torch.utils.data.RandomSampler(trigrams)
    train_trigrams = DataLoader(trigrams, batch_size=BATCH_SIZE, sampler = samp,
                      collate_fn=genbatch)
	
    for epoch in range(5):
        total_loss = 0
        print(epoch)
        for context, target in train_trigrams:
            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in tensors)
            
            # Please uncomment the following lines for KLDiv loss
#            tar = []
#            for i in range(len(target)):
#                vec = np.zeros(len(vocab))
#                vec[target[i]] = 1
#                tar.append(vec)
#            tar = torch.tensor(tar)
            # TODO
            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old instance
            optimizer.zero_grad()
            # TODO
            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            #model = model.float()
            Y = model.forward(context)
            
            # TODO
            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a tensor)
            loss = loss_function(Y, target)
            # TODO
            # assert (0 == 1)
            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()
            # TODO
            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
        losses.append(total_loss)
        print(total_loss)
    print(losses)  # The loss decreased every iteration over the training data!
    # output('./data/embedding.txt_100', './data/embedding_random.txt_100')
    #weight = model.embeddings.weight.data.numpy()
    #weight = numpy.array(weight)
    #print(weight)
    output(model, './data/embedding.txt', './data/random_embedding.txt')

if __name__ == '__main__':
    training()
