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
    

class OnTheFlyMIDIDataset(Dataset):
    def __init__(self, note_list, note_to_int, offset_to_int, duration_to_int, sequence_length):
        self.note_list = note_list
        self.note_to_int = note_to_int
        self.offset_to_int = offset_to_int
        self.duration_to_int = duration_to_int
        self.sequence_length = sequence_length
        # Build an index mapping: (file_idx, start_idx)
        self.indices = []
        for file_idx, file_notes in enumerate(note_list):
            if len(file_notes) > sequence_length:
                for start in range(0, len(file_notes) - sequence_length):
                    self.indices.append((file_idx, start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, start = self.indices[idx]
        file_notes = self.note_list[file_idx]
        seq = file_notes[start:start+self.sequence_length+1]
        pitch_seq = [self.note_to_int[n[0]] for n in seq]
        offset_seq = [self.offset_to_int[round(n[1], 4)] for n in seq]
        duration_seq = [self.duration_to_int[round(n[2], 4)] for n in seq]
        # Input: first N, Output: next N
        return (
            (torch.tensor(pitch_seq[:-1]), torch.tensor(offset_seq[:-1]), torch.tensor(duration_seq[:-1])),
            (torch.tensor(pitch_seq[1:]), torch.tensor(offset_seq[1:]), torch.tensor(duration_seq[1:]))
        )