import torch
import torch.nn as nn

# Following https://colab.research.google.com/github/MITDeepLearning/introtodeeplearning/blob/master/lab1/solutions/PT_Part2_Music_Generation_Solution.ipynb#scrollTo=4HrXTACTdzY- as guide

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, num_offsets, num_durations, embedding_dim, hidden_size, dropout=0.3):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.pitch_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.offset_embedding = nn.Embedding(num_offsets, embedding_dim)
        self.duration_embedding = nn.Embedding(num_durations, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)  # Layer normalization
        self.dropout = nn.Dropout(dropout)  # Dropout applied after LSTM
        self.pitch_head = nn.Linear(hidden_size, vocab_size)  # for pitch classification
        self.offset_head = nn.Linear(hidden_size, num_offsets)  # for offset classification
        self.duration_head = nn.Linear(hidden_size, num_durations)  # for duration classification

    def init_hidden(self, batch_size, device):
        return (torch.zeros(2, batch_size, self.hidden_size).to(device),
                torch.zeros(2, batch_size, self.hidden_size).to(device))

    def forward(self, x, state=None, return_state=False):
        pitch_x = self.pitch_embedding(x[:,:,0])
        offset_x = self.offset_embedding(x[:,:,1])
        duration_x = self.duration_embedding(x[:,:,2])
        x = pitch_x + offset_x + duration_x
        if state is None:
            state = self.init_hidden(x.size(0), x.device)
        out, state = self.lstm(x, state)
        out = self.layer_norm(out)  # Apply layer normalization
        out = self.dropout(out)  # Apply dropout after LSTM
        pitch_logits = self.pitch_head(out)
        offset_logits = self.offset_head(out)
        duration_logits = self.duration_head(out)
        if return_state:
            return (pitch_logits, offset_logits, duration_logits), state
        return pitch_logits, offset_logits, duration_logits
    
"""
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(LSTMModel, self).__init__()

        # Used for initializing hidden layer
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers = 2, batch_first = True) # recurring
        self.fc = nn.Linear(hidden_size, vocab_size) # fully connected
        
        

    def init_hidden(self, batch_size, device):
        return (torch.zeros(1, batch_size, self.hidden_size).to(device), torch.zeros(1, batch_size, self.hidden_size).to(device))
    
    def forward(self, x, state = None, return_state = False):
        x = self.embedding(x)

        if state is None:
            state = self.init_hidden(x.size(0), x.device)
        out, state = self.lstm(x, state)

        out = self.fc(out)
        return out if not return_state else (out, state)
"""


def compute_loss(labels, logits): # basically labels = inputs, logits = outputs
    # Unpack
    pitch_labels, offset_labels, duration_labels = labels
    pitch_logits, offset_logits, duration_logits = logits

    # Flatten for loss computation
    pitch_labels = pitch_labels.view(-1)
    pitch_logits = pitch_logits.view(-1, pitch_logits.size(-1))
    offset_labels = offset_labels.view(-1)
    offset_logits = offset_logits.view(-1, offset_logits.size(-1))
    duration_labels = duration_labels.view(-1)
    duration_logits = duration_logits.view(-1, duration_logits.size(-1))

    # Losses
    pitch_loss = nn.CrossEntropyLoss()(pitch_logits, pitch_labels)
    offset_loss = nn.CrossEntropyLoss()(offset_logits, offset_labels)
    duration_loss = nn.CrossEntropyLoss()(duration_logits, duration_labels)

    # Combine (you can weight them if you want)
    total_loss = pitch_loss + offset_loss + duration_loss
    return total_loss


