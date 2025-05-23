import torch
from torch.utils.data import Dataset

class MIDISequenceDataset(Dataset):
    def __init__(self, input_sequences, output_sequences):
        self.input_sequences = input_sequences
        self.output_sequences = output_sequences

    def __len__(self):
        return len(self.input_sequences)
    
    def __getitem__(self, idx):
        # Each input/output is a tuple: (pitch_seq, offset_seq, duration_se5q)
        pitch_in, offset_in, duration_in = self.input_sequences[idx]
        pitch_out, offset_out, duration_out = self.output_sequences[idx]
        # Convert to tensors
        pitch_in = torch.tensor(pitch_in, dtype=torch.long)
        offset_in = torch.tensor(offset_in, dtype=torch.long)
        duration_in = torch.tensor(duration_in, dtype=torch.long)
        pitch_out = torch.tensor(pitch_out, dtype=torch.long)
        offset_out = torch.tensor(offset_out, dtype=torch.long)
        duration_out = torch.tensor(duration_out, dtype=torch.long)
        return (pitch_in, offset_in, duration_in), (pitch_out, offset_out, duration_out)