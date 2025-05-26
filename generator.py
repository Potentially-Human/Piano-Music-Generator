import torch
from tqdm import tqdm

import torch

def generate_notes(
    model,
    seed_tuple,
    note_to_int, int_to_note,
    offset_to_int, int_to_offset,
    duration_to_int, int_to_duration,
    device,
    length=500
):
    model.eval()
    # Start with the seed (as indices)
    pitch_value, offset_idx, duration_value = seed_tuple
    generated = [(pitch_value, 0, duration_value)]

    # Prepare initial input tensor: shape (1, 1, 3)
    input_seq = torch.tensor([[[note_to_int[pitch_value], offset_idx, duration_to_int[round(duration_value, 4)]]]], dtype=torch.long, device=device)
    state = model.init_hidden(1, device)

    for _ in tqdm(range(length)):
        with torch.no_grad():
            pitch_logits, offset_logits, duration_logits = model(input_seq, state)[0], model(input_seq, state)[1], model(input_seq, state)[2]
            # Take the last time step's output
            pitch_probs = torch.softmax(pitch_logits[:, -1, :], dim=-1)
            offset_probs = torch.softmax(offset_logits[:, -1, :], dim=-1)
            duration_probs = torch.softmax(duration_logits[:, -1, :], dim=-1)

            # Sample from the distributions
            next_pitch_idx = torch.multinomial(pitch_probs, 1).item()
            next_offset_idx = torch.multinomial(offset_probs, 1).item()
            next_duration_idx = torch.multinomial(duration_probs, 1).item()

            # Decode to actual values
            next_pitch = int_to_note[next_pitch_idx]
            next_offset = int_to_offset[next_offset_idx]
            next_duration = int_to_duration[next_duration_idx]

            generated.append((next_pitch, next_offset, next_duration))

            # Prepare next input (append to sequence)
            next_input = torch.tensor([[[next_pitch_idx, next_offset_idx, next_duration_idx]]], dtype=torch.long, device=device)
            input_seq = torch.cat([input_seq, next_input], dim=1)
            
    return generated