import argparse
import time
import math
import os
import os.path
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#from stream_gbw import Vocabulary, StreamGBWDataset
from gbw import GBWDataset
from fast_gbw import FastGBWDataset
import util
import argument

from sparse_model import RNNModel, SampledSoftmax
import transformer as m

from learning_rate import LinearLR
from adam_base import Adam
from rmsprop import RMSProp

parser = argparse.ArgumentParser(description='PyTorch LSTM Language Model')
parser.add_argument('--data', type=str, default='../data/gbw',
                    help='location of the data corpus')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=5,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=20,
                    help='sequence length')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--save', type=str,  default='gbw_model.pt',
                    help='path to save the final model')
#argument.add_recurrent_args(parser)
argument.add_transformer_args(parser)
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

# Torch
word_freq = torch.load(os.path.join(args.data, 'word_freq.pt')).numpy()
mapto = torch.from_numpy(util.reverse(np.argsort(-word_freq))).long()
print("load word frequency mapping - complete")

ntokens = len(word_freq)
nsampled = 8192

train_corpus = FastGBWDataset(args.data, 'train_data.pt', 'train_data.sid', mapto, seq_length=args.bptt, batch_size=args.batch_size)
print("load train data - complete")

test_corpus = GBWDataset(args.data, 'test_data.pt', mapto)
print("load test data - complete")

# Streaming
'''
vocabulary = Vocabulary.from_file(os.path.join(args.data, "1b_word_vocab.txt"))

ntokens = len(vocabulary)
nsampled = 8192

train_corpus = StreamGBWDataset(vocabulary, os.path.join(args.data, "training-monolingual.tokenized.shuffled/*"))
test_corpus = StreamGBWDataset(vocabulary, os.path.join(args.data, "heldout-monolingual.tokenized.shuffled/*"), deterministic=True)
print("load dataset - complete")
'''

###############################################################################
# Build the model
###############################################################################
embed = nn.Embedding(ntokens, args.emsize, sparse=True)
net = m.TransformerModel(m.DecoderPreprocessor(args, embed), m.TransformerDecoder(args, embed))
util.initialize(embed.weight)

twht = None
if args.tied:
    if args.nhid != args.emsize and not args.proj:
        raise ValueError('When using the tied flag, hidden must be equal to embedding size')
    twht = embed.weight

#D = args.emsize if args.proj else args.nhid
D = args.emsize
ss = SampledSoftmax(ntokens, nsampled, D, tied_weight=twht)

net.add_module("embed", embed)
net.add_module("smax", ss)
net.cuda()

print("Batch Size:", args.batch_size, "Initial LR:", args.lr)
criterion = nn.CrossEntropyLoss()
#optimizer = Adam(net.parameters(), args.lr, betas=(0.0, 0.999))
optimizer = RMSProp(net.parameters(), args.lr)
scheduler = LinearLR(optimizer, base_lr=args.lr, max_iters=train_corpus.batch_num*args.epochs, last_iter=-1, min_lr=1e-8)

###############################################################################
# Training code
###############################################################################

def get_batch(item, device_id=0):
    data, target, wrd_cnt, batch_num = item
    return Variable(data.cuda(device_id)), Variable(target.view(-1).cuda(device_id)), wrd_cnt, batch_num

def evaluate(data_source, data_gen):
    # Turn on evaluation mode which disables dropout.
    net.eval()

    total_loss = 0
    total_word_count = 0

    for item in data_gen:
        data, targets, word_cnt, batch_num = get_batch(item)
        output, _ = net(data)
        flat_output = output.view(output.size(0)*output.size(1), output.size(2))
        logits, new_targets = ss(output, targets)

        logits_flat = logits.view(-1, ntokens)
        total_loss += F.cross_entropy(logits_flat, targets, reduction='sum').item()
        batch_size = output.size(0)*output.size(1)
        total_word_count += batch_size
    return total_loss / total_word_count

def train():
    train_loader = train_corpus.batch_generator()
    total_loss = 0
    total_word_count = 0

    start_time = time.time()
    for batch, item in enumerate(train_loader):
        net.train()
        data, targets, word_cnt, batch_len = get_batch(item)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()

        # Network
        output, attn = net(data)
        flat_output = output.view(output.size(0)*output.size(1), output.size(2))
        logits, new_targets = ss(flat_output, targets)

        loss = criterion(logits.view(-1, nsampled+1), new_targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        torch.nn.utils.clip_grad_norm_(embed.parameters(), args.clip)
        torch.nn.utils.clip_grad_norm_(ss.parameters(), args.clip)

        optimizer.step()
        scheduler.step()

        total_loss += word_cnt * loss.data
        total_word_count += word_cnt

        interval = max(10, 125)
        if (batch % interval) == 0:
            elapsed = time.time() - start_time
            print('Epoch: {:3d} | {:5d}/{:5d} batches | lr {:.6f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'
                  .format(epoch, batch, batch_len, scheduler.lr, elapsed * 1000 / interval, loss.item(), math.exp(loss.item())))
            start_time = time.time()
            sys.stdout.flush()

# Load the saved model.
if os.path.isfile(args.save):
    print("Loading Saved Model")
    with open(args.save, 'rb') as f:
        net.load_state_dict(torch.load(f))
else:
    print("Random Initialization - No Saved Model")

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        with open(args.save, 'wb') as f:
             torch.save(net.state_dict(), f)

        test_loader = test_corpus.batch_generator(seq_length=args.bptt, batch_size=32, shuffle=False)
        val_loss = evaluate(test_corpus, test_loader)
        print('-' * 89)
        print('Test: {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'
               .format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
        print('-' * 89)
        sys.stdout.flush()
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
    sys.stdout.flush()

# Run on test data.
test_loader = test_corpus.batch_generator(seq_length=args.bptt, batch_size=32, shuffle=False)
test_loss = evaluate(test_corpus, test_loader)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('=' * 89)
