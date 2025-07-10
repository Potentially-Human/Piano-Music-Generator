from music21 import converter, instrument, note, chord, stream, tempo
from midi_parsing import parse_midi, write_midi
import pickle
import glob
import numpy as np
from itertools import chain
import torch
from tqdm import tqdm
from math import floor

def notes(path: str, sequence_length, ending_num) -> list:
    note_list = []
    end_note_list = []
    combined_note_list = []
    for file in tqdm(glob.glob(path + "/*.mid")):
        try: 
            file_note_list = parse_midi(file)
            combined_note_list.append(file_note_list)

            if len(file_note_list) <= sequence_length + ending_num:
                note_list.append(file_note_list)
            else:
                note_list.append(file_note_list[:-ending_num])
                end_note_list.append(file_note_list[-ending_num-sequence_length:] + [("Terminate", 0, 0, 0)])
        except:
            print(file)
            continue

    return note_list, end_note_list, combined_note_list

def get_vocab(combined_note_list: list) -> tuple:
    # Flatten all (pitch, offset, duration) tuples across all files
    all_pitches = set()
    for file in combined_note_list:
        for (pitch, offset, duration, volume) in file:
            all_pitches.add(pitch)
            pitch_strings = []
    pitch_ints = []
    for i in all_pitches:
        if type(i) == type(1):
            pitch_ints.append(i)
        else:
            pitch_strings.append(i)
    pitch_strings = sorted(pitch_strings)
    pitch_ints = sorted(pitch_ints)
    pitch_vocab = pitch_ints + pitch_strings
    pitch_vocab.append("Terminate")
    
    return (
        {u: i for i, u in enumerate(pitch_vocab)}, pitch_vocab
    )





# The rest of this file is unused after the current update. 

# attempting to open midi files

# https://github.com/arman-aminian/lofi-generator/blob/master/Piano%20Generator.ipynb

def standardize_duration(f: float) -> float:
    valid_fractions = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.6666666666666666, 0.5, 0.375, 0.3333333333333333, 0.25, 0.1875, 0.16666666666666666, 0.125, 0.09375, 0.08333333333333333, 0.0625, 0.046875, 0.041666666666666664, 0.03125, 0.0234375, 0.020833333333333332, 0.015625, 0.01171875, 0.010416666666666666, 0.0078125, 0.005859375, 0.005208333333333333, 0.00390625, 0.0029296875, 0.0026041666666666665, 0.001953125, 0.00146484375, 0.0013020833333333333, 0.0009765625, 0.000732421875, 0.0006510416666666666, 0.00048828125, 0.0003662109375, 0.0003255208333333333, 0.000244140625, 0.00018310546875, 0.00016276041666666666, 0.0001220703125, 9.1552734375e-05, 8.138020833333333e-05, 6.103515625e-05, 4.57763671875e-05, 4.0690104166666664e-05, 3.0517578125e-05, 2.288818359375e-05, 2.0345052083333332e-05, 1.52587890625e-05, 1.1444091796875e-05, 1.0172526041666666e-05, 7.62939453125e-06, 5.7220458984375e-06, 5.086263020833333e-06, 3.814697265625e-06, 2.5431315104166665e-06, 1.9073486328125e-06, 1.2715657552083333e-06, 0]
    return floor(f) + round(min(valid_fractions, key = lambda x: abs(x - (f % 1))), 4)
    


# OK I kinda get this part and it isn't purely plagerized anymore, just mostly plagerized

"""

def notes(path : str) -> list:
    # list to store notes
    note_list = []
    for file in glob.glob(path + "/*.mid"):
        midi = converter.parse(file)

        # Basically these are notes that need to be looked at
        notes_to_parse = None

        try:
            notes_to_parse = instrument.partitionByInstrument("midi")
        except:
            notes_to_parse = midi.flatten().notes
        
        file_notes = []
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                file_notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                file_notes.append('.'.join(str(n) for n in element.normalOrder))
        
        note_list.append(file_notes)

    with open('data/notes', 'wb') as filepath:
        pickle.dump(note_list, filepath)

    return note_list
"""

def notes_old(path: str, sequence_length, ending_num) -> list:
    # List to store notes with their offsets, durations, and tempo
    note_list = []
    end_note_list = []
    combined_note_list = []
    for file in tqdm(glob.glob(path + "/*.mid")):
        midi = None
        try:
            midi = converter.parse(file, quantizePost=False)
        except:
            print("Invalid file: " + file)
            continue
        # Extract tempo
        try:
            tempo = midi.flatten().getElementsByClass('MetronomeMark')[0].number
        except IndexError:
            tempo = 120  # Default tempo if none is found
        
        tempo_factor = 120 / tempo
        notes_to_parse = midi.flatten().notes

        # Extract pedal information
        """ pedals = []

        for element in midi.flatten().getElementsByClass('ControlChange'):
            print(element.number)
            if element.number == 64:  # Sustain pedal (MIDI CC 64)
                pedals.append((round(float(element.offset), 5), element.value))"""

        pedals_gone_through = 0

        last_note_offset = 0

        add_ending = len(notes_to_parse) > sequence_length + ending_num 

        file_notes = []
        end_notes = []
        combined_notes = []

        """try:
            if pedals[0][0] == 0 and pedals[0][1] == 0:
                file_notes.append("pedalDown/0/NA")
                pedals_gone_through += 1
        except IndexError:
            pass"""

        for i, element in enumerate(notes_to_parse):
            """
            try:
                for _ in range(2):
                    if pedals[pedals_gone_through][0] > last_note_offset and pedals[pedals_gone_through][0] <= element.offset:
                        pedal = str(pedals[pedals_gone_through][0]) + "/NA"
                        if last_note_offset:
                            pedal = "pedalUp/" + pedal
                        else:
                            pedal = "pedalDown/" + pedal
                        file_notes.append(pedal)
            except IndexError:
                pass
"""
            offset = tempo_factor * float(element.offset)
            duration = tempo_factor * float(element.quarterLength)

            if isinstance(element, note.Note):
                combined_notes.append(
                    (str(element.pitch), standardize_duration(offset - last_note_offset), standardize_duration(duration))
                )
            elif isinstance(element, chord.Chord):
                # if float(element.quarterLength) == 0:
                #     continue
                # Extract chord, offset, and duration
                combined_notes.append(
                    ('.'.join(str(n) for n in element.normalOrder), standardize_duration(offset - last_note_offset), standardize_duration(duration))
                )

            if add_ending and i >= len(notes_to_parse) - sequence_length - ending_num:
                if isinstance(element, note.Note):
                    end_notes.append(
                        (str(element.pitch), standardize_duration(offset - last_note_offset), standardize_duration(duration))
                    )
                elif isinstance(element, chord.Chord):
                    # if float(element.quarterLength) == 0:
                    #     continue
                    # Extract chord, offset, and duration
                    end_notes.append(
                        ('.'.join(str(n) for n in element.normalOrder), standardize_duration(offset - last_note_offset), standardize_duration(duration))
                    )
                last_note_offset = offset
                continue

            if isinstance(element, note.Note):
                
                # if float(element.quarterLength) == 0:
                #     continue
                # Extract pitch, offset, and duration
                file_notes.append(
                    (str(element.pitch), standardize_duration(offset - last_note_offset), standardize_duration(duration))
                )
            elif isinstance(element, chord.Chord):
                # if float(element.quarterLength) == 0:
                #     continue
                # Extract chord, offset, and duration
                file_notes.append(
                    ('.'.join(str(n) for n in element.normalOrder), standardize_duration(offset - last_note_offset), standardize_duration(duration))
                )
            last_note_offset = offset

        note_list.append(file_notes)
        end_notes.append(("Terminate", 0, 0))
        end_note_list.append(end_notes)
        combined_note_list.append(combined_notes)

    # Save the notes with offsets, durations, and tempo
    """with open('data/notes', 'wb') as filepath:
        pickle.dump(note_list, filepath)"""

    return note_list, end_note_list, combined_note_list
    


# Yeah ok I just plagerized this function
# Or used Mr. GPT
"""
def create_midi(prediction_output, 
                output_path):
    # convert the output from the prediction to notes and create a midi file from the notes
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
#         offset += 0.5
        offset += 1

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp=output_path)
"""

def create_midi_old(prediction_output, output_path):
    """Convert the output from the prediction to notes and create a MIDI file
       from the notes, including offset, duration, and tempo."""

    # Extract tempo from the first element (assuming tempo is consistent)
    tempo_mark = tempo.MetronomeMark(number=120)

    # Add tempo to the MIDI stream
    midi_stream = stream.Stream()
    midi_stream.append(tempo_mark)

    pedals = []

    last_note_offset = 0

    # Create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        pitch, offset, duration = pattern

        offset, duration = last_note_offset + float(offset), float(duration)
        """
        if pitch == "pedalDown":
            pedals.append((offset, 0))
            continue
        elif pitch == "pedalUp":
            pedals.append((offset, 127))
            continue"""
        
        if ('.' in pitch) or pitch.isdigit():
            # Create a chord object
            notes_in_chord = pitch.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset  # Use the exact offset for simultaneous playback
            new_chord.quarterLength = duration
            midi_stream.insert(offset, new_chord)  # Use insert instead of append
        else:
            # Create a note object
            new_note = note.Note(pitch)
            new_note.offset = offset  # Use the exact offset for simultaneous playback
            new_note.quarterLength = duration
            new_note.storedInstrument = instrument.Piano()
            midi_stream.insert(offset, new_note)  # Use insert instead of append

        last_note_offset = offset


    for pedal in pedals:
        pedal_event = stream.Stream()
        pedal_event.append(tempo.MetronomeMark(number=pedal[1]))
        midi_stream.insert(pedal[0], pedal_event)

    # Write the MIDI file
    midi_stream.write('midi', fp=output_path)



""""

The following functions are unused and are not updated 

"""

# No longer used
def generate_io_sequences(note_list, note_to_int, offset_to_int, duration_to_int, sequence_length=300):
    input_seq = []
    output_seq = []

    for file_note in note_list:
        for i in range(0, len(file_note) - sequence_length):
            seq_in = file_note[i:i + sequence_length]
            seq_out = file_note[i + 1 : i + sequence_length + 1]

            # Always round offsets and durations to 6 decimals before lookup
            pitch_in = torch.tensor([note_to_int[n[0]] for n in seq_in])
            offset_in = torch.tensor([offset_to_int[round(float(n[1]), 4)] for n in seq_in])
            duration_in = torch.tensor([duration_to_int[round(float(n[2]), 4)] for n in seq_in])

            pitch_out = torch.tensor([note_to_int[n[0]] for n in seq_out])
            offset_out = torch.tensor([offset_to_int[round(float(n[1]), 4)] for n in seq_out])
            duration_out = torch.tensor([duration_to_int[round(float(n[2]), 4)] for n in seq_out])

            input_seq.append((pitch_in, offset_in, duration_in))
            output_seq.append((pitch_out, offset_out, duration_out))

    return input_seq, output_seq


# No longer used
def get_batch(input_sequences, output_sequences, batch_size):
    idx = np.random.choice(len(input_sequences), batch_size)
    input_batch = [input_sequences[i] for i in idx]
    output_batch = [output_sequences[i] for i in idx]

    # Each input/output is a tuple: (pitch_seq, offset_seq, duration_seq)
    pitch_in = torch.tensor([x[0] for x in input_batch], dtype=torch.long)
    offset_in = torch.tensor([x[1] for x in input_batch], dtype=torch.long)
    duration_in = torch.tensor([x[2] for x in input_batch], dtype=torch.long)

    pitch_out = torch.tensor([x[0] for x in output_batch], dtype=torch.long)
    offset_out = torch.tensor([x[1] for x in output_batch], dtype=torch.long)
    duration_out = torch.tensor([x[2] for x in output_batch], dtype=torch.long)

    # Return as tuples for model and loss
    return (pitch_in, offset_in, duration_in), (pitch_out, offset_out, duration_out)


# Unused, note_list should take combined_note_list
def end_notes(note_list, sequence_length, ending_num):
    ending_note_list = []
    for element in note_list:
        if len(element) <= sequence_length + ending_num:
            end = element.copy()
            end.append(("Terminate", 0, 0))
            ending_note_list.append(end)
        else:
            end = element[-sequence_length - ending_num:-1].copy()
            end.append(("Terminate", 0, 0))
            ending_note_list.append(end)

    return ending_note_list

        