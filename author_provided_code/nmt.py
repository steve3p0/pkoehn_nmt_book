import torch

from io import open
import unicodedata
import string
import re
import random

MAX_LENGTH=100

class Vocabulary:
    def __init__(self):
        self.word2index = {"<s>": 0, "</s>": 1}
        self.index2word = {0: "<s>", 1: "</s>"}
        self.n_words = 2  # Count SOS and EOS

    def getIndex(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
        return self.word2index[word]

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readCorpus(file):
    # Read the file and split into lines
    lines = open(file, encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    text_corpus = ["<s> " + normalizeString(l) + " </s>" for l in lines]

    return text_corpus

class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, max_length=MAX_LENGTH):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(torch.tensor([[input]])).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def process_sentence(self, input_sentence):
        hidden = torch.zeros(1, 1, self.hidden_size)
        encoder_output = torch.zeros(self.max_length, self.hidden_size)
        for i in range(len(input_sentence)):
            embedded = self.embedding(torch.tensor([[input_sentence[i]]])).view(1, 1, -1)
            output, hidden = self.gru(embedded, hidden)
            encoder_output[i] = output[0, 0]
        return encoder_output

class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, max_length=MAX_LENGTH):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_length = max_length

        self.gru = torch.nn.GRU(2 * hidden_size, hidden_size)

        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.Wa = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ua = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.va = torch.nn.Parameter(torch.FloatTensor(1,hidden_size))

        self.out = torch.nn.Linear(3 * hidden_size, vocab_size)

    def forward(self, prev_output_id, prev_hidden, encoder_output, input_length):
        prev_output = self.embedding(torch.tensor([prev_output_id])).unsqueeze(1)

        m = torch.tanh(self.Wa(prev_hidden) + self.Ua(encoder_output))
        attention_scores = m.bmm(self.va.unsqueeze(2)).squeeze(-1)
        attention_scores = self.mask(attention_scores, input_length)
        attention_weights = torch.nn.functional.softmax( attention_scores, -1 )

        context = attention_weights.unsqueeze(1).bmm(encoder_output.unsqueeze(0))

        rnn_input = torch.cat((prev_output, prev_hidden), 2)
        rnn_output, hidden = self.gru(rnn_input, prev_hidden)

        output = self.out(torch.cat((rnn_output, context, prev_output), 2))
        output = torch.nn.functional.log_softmax(output[0], dim=1)
        return output, hidden

    def mask(self, scores, input_length):
        s = scores.squeeze(0)
        for i in range(self.max_length-input_length):
            s[input_length+i] = -float('inf')
        return s.unsqueeze(0)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

source_text_corpus = readCorpus("Tanzil.20k.de-en.de")
target_text_corpus = readCorpus("Tanzil.20k.de-en.en")
source_vocab = Vocabulary()
target_vocab = Vocabulary()
source_numbered_corpus = [[source_vocab.getIndex(word) for word in sentence.split(' ')] for sentence in source_text_corpus]
target_numbered_corpus = [[target_vocab.getIndex(word) for word in sentence.split(' ')] for sentence in target_text_corpus]

criterion = torch.nn.NLLLoss()

hidden_size = 256
learning_rate = 0.01

encoder = Encoder(source_vocab.n_words, hidden_size)
decoder = Decoder(target_vocab.n_words, hidden_size)

encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)

for epoch in range(100):
    total_loss = 0
    for source_sentence, target_sentence in zip(source_numbered_corpus, target_numbered_corpus):
        if len(source_sentence) > MAX_LENGTH:
            continue
        if len(target_sentence) > MAX_LENGTH:
            continue

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_output = encoder.process_sentence( source_sentence )

        sentence_loss = 0
        hidden = decoder.initHidden()
        for i in range(len(target_sentence)-1):
            output, hidden = decoder.forward( target_sentence[i], hidden, encoder_output, len(source_sentence) )
            word_loss = criterion( output, torch.tensor([target_sentence[i+1]]))
            print("correct: %15s, loss: %5.2f, prediction: %s" % (target_vocab.index2word[target_sentence[i+1]], word_loss, target_vocab.index2word[torch.argmax(output, dim=1).item()]))
            sentence_loss += word_loss
        #print(loss.data.item())
        total_loss += sentence_loss.data.item();
        sentence_loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

    print(total_loss / len(numbered_corpus))


