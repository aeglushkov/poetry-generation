import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tqdm

from weight_drop import WeightDrop


def to_matrix(sentences, token_to_idx, max_len=None, dtype='int32', batch_first=True):
    """Casts a list of sentences into rnn-digestable matrix"""
    
    pad = token_to_idx[' ']
    
    max_len = max_len or max(map(len, sentences))
    sentences_ix = np.zeros([len(sentences), max_len], dtype) + pad

    for i in range(len(sentences)):
        line_ix = [token_to_idx[c] for c in sentences[i]]
        sentences_ix[i, :len(line_ix)] = line_ix[:max_len]
        
    if not batch_first:
        sentences_ix = np.transpose(sentences_ix)

    return sentences_ix


class CharRNNCell(nn.Module):
    def __init__(self, num_tokens, emb_size=32, rnn_num_units=256, emb_dropout=0):
        super(self.__class__, self).__init__()
        self.num_units = rnn_num_units
        
        self.emb = nn.Embedding(num_tokens, emb_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.rnn = nn.Linear(emb_size + rnn_num_units, rnn_num_units)
        self.hid_to_logits = nn.Linear(rnn_num_units, num_tokens)
        
    def forward(self, x, h_prev):
        x_emb = self.emb(x)
        x_emb = self.emb_dropout(x_emb)
        x_and_h = torch.cat([x_emb, h_prev], dim=-1)
        h_next = self.rnn(x_and_h)
        
        next_logits = self.hid_to_logits(h_next)
        next_logp = F.log_softmax(next_logits, dim=-1)
        return h_next, next_logp

    def initial_state(self, batch_size):
        """ return rnn state before it processes first input (aka h0) """
        return torch.zeros(batch_size, self.num_units, requires_grad=True)

    
def rnn_loop(char_rnn, batch_ix, device):
    """
    Computes log P(next_character) for all time-steps in names_ix
    :param names_ix: an int32 matrix of shape [batch, time], output of to_matrix(names)
    """
    batch_size, max_length = batch_ix.size()
    hid_state = char_rnn.initial_state(batch_size).to(device)
    logprobs = []

    for x_t in batch_ix.transpose(0, 1):
        hid_state, logp_next = char_rnn(x_t, hid_state)
        logprobs.append(logp_next)
        
    return torch.stack(logprobs, dim=1)


def train_char_rnn(
    char_rnn, 
    poems, 
    criterion, 
    opt, 
    scheduler, 
    history, 
    token_to_idx,
    num_tokens, 
    device="cpu", 
    epochs=100, 
    batch_size=32
):
    char_rnn.train()
    char_rnn.to(device)
    loss_avg = []
    poems_dataloader = DataLoader(poems, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
    for epoch in tqdm.tqdm(range(epochs), desc="training..."):
        epoch_history = []
        for batch in poems_dataloader:
            opt.zero_grad()
            
            batch_ix = to_matrix(batch, token_to_idx, max_len=None)
            batch_ix = torch.tensor(batch_ix, dtype=torch.int64).to(device)

            logp_seq = rnn_loop(char_rnn, batch_ix, device)

            # compute loss
            predictions_logp = logp_seq[:, :-1]
            actual_next_tokens = batch_ix[:, 1:]

            loss = criterion(predictions_logp.contiguous().view(-1, num_tokens), 
                             actual_next_tokens.contiguous().view(-1))

            # train with backprop
            loss.backward()
            opt.step()

            epoch_history.append(loss.cpu().data.numpy())
            
        history.append(np.mean(epoch_history))
        
        loss_avg.append(history[-1])
        if len(loss_avg) >= 10:
            scheduler.step(np.mean(loss_avg))
            loss_avg = []
            
        if (epoch + 1) % 10 == 0:        
            clear_output(True)
            plt.plot(history, label='loss')
            plt.legend()
            plt.show()
            
            
class LSTM(nn.Module):
    def __init__(
        self, 
        num_tokens, 
        emb_size=32, 
        rnn_num_units=256, 
        n_layers=1, 
        emb_dropout=0, 
        lstm_dropout=0, 
        weight_dropout=None
    ):
        super(self.__class__, self).__init__()
        self.num_units = rnn_num_units
        self.n_layers = n_layers
        self.lstm_dropout = lstm_dropout
        
        self.emb = nn.Embedding(num_tokens, emb_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.lstm = nn.LSTM(emb_size, rnn_num_units, n_layers, batch_first=True, dropout=lstm_dropout)
        if weight_dropout:
            self.lstm = WeightDrop(self.lstm, ['weight_hh_l0'], dropout=weight_dropout)
        self.hid_to_logits = nn.Linear(rnn_num_units, num_tokens)
        
    def forward(self, x, hidden):
        x_emb = self.emb(x)
        x_emb = self.emb_dropout(x_emb)
        out, hidden = self.lstm(x_emb, hidden)
        next_logits = self.hid_to_logits(out)
        next_logp = F.log_softmax(next_logits, dim=-1)
        return next_logp, hidden
    
    def initial_state(self, batch_size=1, device="cpu"):
        """ return rnn state before it processes first input (aka h0) """
        return (torch.zeros(self.n_layers, batch_size, self.num_units, requires_grad=True).to(device),
                torch.zeros(self.n_layers, batch_size, self.num_units, requires_grad=True).to(device))
    
    
def train_lstm(
    lstm_model, 
    poems, 
    criterion, 
    opt, 
    scheduler, 
    history, 
    token_to_idx, 
    num_tokens, 
    device="cpu", 
    epochs=100, 
    batch_size=32
):
    lstm_model.train()
    lstm_model.to(device)
    loss_avg = []
    poems_dataloader = DataLoader(poems, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
    for epoch in tqdm.tqdm(range(epochs), total=epochs, desc="training..."):
        epoch_history = []
        for batch in poems_dataloader:
            opt.zero_grad()

            batch_ix = to_matrix(batch, token_to_idx, max_len=None)
            batch_ix = torch.tensor(batch_ix, dtype=torch.long).to(device)
            hidden = lstm_model.initial_state(batch_size, device)
            
            logp_seq, hidden = lstm_model(batch_ix, hidden)
            # compute loss
            loss = criterion(logp_seq[:, :-1].cpu().contiguous().view(-1, num_tokens), 
                             batch_ix[:, 1:].cpu().contiguous().view(-1))
            # train with backprop
            loss.backward()
            opt.step()
            
            epoch_history.append(loss.data.numpy())
            
        history.append(np.mean(epoch_history))
        
        loss_avg.append(history[-1])
        if len(loss_avg) >= 10:
            scheduler.step(np.mean(loss_avg))
            loss_avg = []
            
        if (epoch + 1) % 10 == 0:        
            clear_output(True)
            plt.plot(history, label='loss')
            plt.legend()
            plt.show()