import os
import random
import numpy as np
import math
import torch
from torch import nn
from torch.nn import init
from torch import optim
import torch.nn.functional as F


# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html

class attention_model(nn.Module):

    def __init__(self, vocab_size, hidden_dim=64, attention_type="none"):
        super(attention_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeds = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim)
        self.decoder = nn.LSTM(2*hidden_dim, hidden_dim) if attention_type != "none" \
                                                         else nn.LSTM(hidden_dim, hidden_dim)
        self.loss = nn.CrossEntropyLoss()
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.v = torch.rand(hidden_dim, 1)
        self.w1 = torch.rand(hidden_dim, hidden_dim)
        self.w2 = torch.rand(hidden_dim, hidden_dim)
        self.attention_type = attention_type

        # Attention
        self.attn = nn.Linear(self.hidden_dim * 2, hidden_dim)
        self.attn_combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

    def compute_Loss(self, pred_vec, gold_seq):
        return self.loss(pred_vec, gold_seq)

    def forward(self, input_seq, gold_seq=None):
        input_vectors = self.embeds(torch.tensor(input_seq))
        input_vectors = input_vectors.unsqueeze(1)
        enc_outputs, hidden = self.encoder(input_vectors)
        outputs = enc_outputs

        # Technique used to train RNNs:
        # https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/
        teacher_force = False

        # This condition tells us whether we are in training or inference phase
        if gold_seq is not None and teacher_force:
            gold_vectors = torch.cat([torch.zeros(1, self.hidden_dim), self.embeds(torch.tensor(gold_seq[:-1]))], 0)
            gold_vectors = gold_vectors.unsqueeze(1)
            gold_vectors = torch.nn.functional.relu(gold_vectors)
            outputs, hidden = self.decoder(gold_vectors, hidden)
            predictions = self.out(outputs)
            predictions = predictions.squeeze()
            vals, idxs = torch.max(predictions, 1)
            return predictions, list(np.array(idxs))
        else:
            prev = torch.zeros(1, 1, self.hidden_dim)
            predictions = []
            predicted_seq = []
            for j in range(len(input_seq)):
                prev = torch.nn.functional.relu(prev)

                if self.attention_type != "none":
                    # calculate a_j vector
                    a = torch.zeros(len(input_seq))
                    for i in range(len(input_seq)):
                        # for each i in [1, N]:
                        # calculate a_i
                        if self.attention_type == "add":
                            a[i] = self.additive(hidden[0], enc_outputs[i])
                        else:
                            # mult
                            a[i] = self.multiplicative(hidden[0], enc_outputs[i])
                    # a_j = [a_1 ... a_n=N]
                    a = F.softmax(a)

                    # calculate context vector c
                    c = torch.zeros(1, self.hidden_dim)
                    for i in range(len(input_seq)):
                        c += a[i].unsqueeze(0) * enc_outputs[i]

                    # decode_input = concatenate prev, c
                    decode_input = torch.cat((prev, c.unsqueeze(1)), 2)

                    outputs, hidden = self.decoder(decode_input, hidden)
                else:
                    # no attention
                    outputs, hidden = self.decoder(prev, hidden)

                pred_i = self.out(outputs)
                pred_i = pred_i.squeeze()
                _, idx = torch.max(pred_i, 0)
                idx = idx.item()
                predictions.append(pred_i)
                predicted_seq.append(idx)
                prev = self.embeds(torch.tensor([idx]))
                prev = prev.unsqueeze(1)
            return torch.stack(predictions), predicted_seq

    def additive(self, hidden, e):
        # vT.Relu (w1.h + w2.e)
        hidden = hidden.squeeze(1)
        v_transpose = torch.t(self.v)
        #addition = torch.mm(self.w1, torch.t(hidden)).add(torch.mm(self.w2, torch.t(e)))
        addition = torch.mm(self.w1, torch.t(hidden)) + (torch.mm(self.w2, torch.t(e)))
        #reluOutput = F.relu(addition)
        reluOutput = torch.tanh(addition)
        #reluOutput = addition
        finalOutput = torch.dot(v_transpose.squeeze(), reluOutput.squeeze())
        return finalOutput

    def multiplicative(self, hidden, e):
        # vT.Relu (w1.h + w2.e)
        hidden = hidden.squeeze(1)
        finalOutput = torch.mm(torch.mm(hidden, self.w1), torch.t(e))
        return finalOutput
