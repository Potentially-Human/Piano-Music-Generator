import torch
import torch.nn as nn

# Following https://colab.research.google.com/github/MITDeepLearning/introtodeeplearning/blob/master/lab1/solutions/PT_Part2_Music_Generation_Solution.ipynb#scrollTo=4HrXTACTdzY- as guide

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, max_offsets, max_durations, max_volumes, embedding_dim, hidden_size, dropout=0.3):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.pitch_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.offset_embedding = nn.Embedding(max_offsets + 1, embedding_dim)
        self.duration_embedding = nn.Embedding(max_durations + 1, embedding_dim)
        self.volume_embedding = nn.Embedding(max_volumes + 1, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)  # Layer normalization
        self.dropout = nn.Dropout(dropout)  # Dropout applied after LSTM
        self.pitch_head = nn.Linear(hidden_size, vocab_size)  # for pitch classification
        self.offset_head = nn.Linear(hidden_size, max_offsets + 1)  # for offset classification
        self.duration_head = nn.Linear(hidden_size, max_durations + 1)  # for duration classification
        self.volume_head = nn.Linear(hidden_size, max_volumes + 1)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(2, batch_size, self.hidden_size).to(device),
                torch.zeros(2, batch_size, self.hidden_size).to(device))

    def forward(self, x, state=None, return_state=False):
        pitch_x = self.pitch_embedding(x[:,:,0])
        offset_x = self.offset_embedding(x[:,:,1])
        duration_x = self.duration_embedding(x[:,:,2])
        volume_x = self.volume_embedding(x[:,:,3])
        x = pitch_x + offset_x + duration_x + volume_x
        if state is None:
            state = self.init_hidden(x.size(0), x.device)
        out, state = self.lstm(x, state)
        out = self.layer_norm(out)  # Apply layer normalization
        out = self.dropout(out)  # Apply dropout after LSTM
        pitch_logits = self.pitch_head(out)
        offset_logits = self.offset_head(out)
        duration_logits = self.duration_head(out)
        volume_logits = self.volume_head(out)
        if return_state:
            return (pitch_logits, offset_logits, duration_logits, volume_logits), state
        return pitch_logits, offset_logits, duration_logits, volume_logits
    
"""
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(LSTMModel, self).__init__()

        # Used for initializing hidden layer
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, max_layers = 2, batch_first = True) # recurring
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


def compute_loss(labels, logits, max_offsets, max_durations, max_volumes, device): # basically labels = inputs, logits = outputs
    # Unpack
    pitch_labels, offset_labels, duration_labels, volume_labels = labels
    pitch_logits, offset_logits, duration_logits, volume_logits = logits

    # Flatten for loss computation
    pitch_labels, offset_labels, duration_labels, volume_labels = pitch_labels.view(-1), offset_labels.view(-1), duration_labels.view(-1), volume_labels.view(-1)
    pitch_logits, offset_logits, duration_logits, volume_logits = pitch_logits.view(-1, pitch_logits.size(-1)), offset_logits.view(-1, offset_logits.size(-1)), duration_logits.view(-1, duration_logits.size(-1)), volume_logits.view(-1, volume_logits.size(-1))

    # Losses
    pitch_loss = nn.CrossEntropyLoss()(pitch_logits, pitch_labels)
    """offset_probs = torch.softmax(offset_logits, dim = -1)
    offset_pred_value = (offset_probs * torch.arange(max_offsets + 1, device = device)).sum(dim=-1)
    offset_loss = nn.MSELoss()(offset_pred_value, offset_labels.float())
    duration_probs = torch.softmax(duration_logits, dim = -1)
    duration_pred_value = (duration_probs * torch.arange(max_durations + 1, device = device)).sum(dim=-1)
    duration_loss = nn.MSELoss()(duration_pred_value, duration_labels.float())"""
    volume_probs = torch.softmax(volume_logits, dim = -1)
    volume_pred_value = (volume_probs * torch.arange(max_volumes + 1, device = device)).sum(dim=-1)
    volume_loss = nn.MSELoss()(volume_pred_value, volume_labels.float())

    offset_loss = nn.CrossEntropyLoss()(offset_logits, offset_labels)
    duration_loss = nn.CrossEntropyLoss()(duration_logits, duration_labels)

    # Combine (you can weight them if you want)
    total_loss = 4 * pitch_loss + 2 * offset_loss + 2 * duration_loss + volume_loss
    return total_loss


