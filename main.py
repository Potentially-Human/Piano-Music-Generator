from dataparser import notes, get_vocab, generate_io_sequences, get_batch, create_midi
from model import LSTMModel, compute_loss
from trainer import train_step
import torch
from tqdm import tqdm
from generator import generate_notes
from torch.utils.data import DataLoader
from dataset import MIDISequenceDataset, OnTheFlyMIDIDataset
#from midi2audio import FluidSynth
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt




def main(trained = False, on_the_fly = True) -> None:

    count = 5

    params = dict(
        embedding_dim = 512,
        hidden_size = 2048,
        batch_size = 16,
        sequence_length = 300,
        learning_rate = 1e-3,
        training_iterations = 3000,
        output_length = 500,
    )




    path = "midi_files"
    model_dir = "./models"

    model = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not trained:
        note_list = notes(path)
        note_to_int, int_to_note, offset_to_int, int_to_offset, duration_to_int, int_to_duration = get_vocab(note_list)

        note_dictionary = set()
        for midi_file in note_list:
            for note in midi_file:
                note_dictionary.add(note)

        note_dictionary = list(note_dictionary)


        vocabs = {
            "note_to_int": note_to_int,
            "int_to_note": int_to_note,
            "offset_to_int": offset_to_int,
            "int_to_offset": int_to_offset,
            "duration_to_int": duration_to_int,
            "int_to_duration": int_to_duration,
            "note_dictionary": note_dictionary,
        }

        with open(os.path.join(model_dir, "vocabs.pkl"), "wb") as f:
            pickle.dump(vocabs, f)

        vocab_size = len(int_to_note)
        offset_types = len(int_to_offset)
        duration_types = len(int_to_duration)


        dataset = None
        if not on_the_fly:
            input_sequences, output_sequences = generate_io_sequences(note_list, note_to_int, offset_to_int, duration_to_int, sequence_length = params["sequence_length"])
            dataset = MIDISequenceDataset(input_sequences, output_sequences)
        else: 
            dataset = OnTheFlyMIDIDataset(note_list, note_to_int, offset_to_int, duration_to_int, params["sequence_length"])
        dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True, drop_last=True)

        # I guess this decides what the training and model runs on
        model = LSTMModel(vocab_size, offset_types, duration_types, params["embedding_dim"], params["hidden_size"]).to(device)
        

        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

        # print(model(input_sequences[0 : 5]))
        
        os.makedirs(model_dir, exist_ok = True)

        # Training Process

        if hasattr(tqdm, '_instances'): tqdm._instances.clear() # tqdm is the progress bar thing

        losses = []

        for it, (x_vals, y_vals) in enumerate(tqdm(dataloader)):
            if it >= params["training_iterations"]:
                break
            elif it % 100 == 0:
                torch.save(model.state_dict(), os.path.join(model_dir, "mdl.pth"))
            x_pitch, x_offset, x_duration = x_vals
            y_pitch, y_offset, y_duration = y_vals
            x_pitch, x_offset, x_duration = x_pitch.to(device), x_offset.to(device), x_duration.to(device)
            y_pitch, y_offset, y_duration = y_pitch.to(device), y_offset.to(device), y_duration.to(device)

            # Stack inputs along last dimension for embedding
            x = torch.stack([x_pitch, x_offset, x_duration], dim=2)  # shape: (batch, seq, 3)
            y = (y_pitch, y_offset, y_duration)

            loss = train_step(x, y, model, optimizer, compute_loss)
            losses.append(loss)
        torch.save(model.state_dict(), os.path.join(model_dir, "mdl.pth"))
        
        plt.plot(losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.savefig(os.path.join(model_dir, "loss_curve.png"))
    else:
        with open(os.path.join(model_dir, "vocabs.pkl"), "rb") as f:
            vocabs = pickle.load(f)
        note_to_int = vocabs["note_to_int"]
        int_to_note = vocabs["int_to_note"]
        offset_to_int = vocabs["offset_to_int"]
        int_to_offset = vocabs["int_to_offset"]
        duration_to_int = vocabs["duration_to_int"]
        int_to_duration = vocabs["int_to_duration"]
        note_dictionary = vocabs["note_dictionary"]

        # print(note_dictionary)
        print(duration_to_int)

        model = LSTMModel(
            vocab_size=len(int_to_note),
            num_offsets=len(int_to_offset),
            num_durations=len(int_to_duration),
            embedding_dim=params["embedding_dim"],
            hidden_size=params["hidden_size"]
        ).to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, "mdl.pth"), map_location = device))
        model.eval()
    os.makedirs("./output", exist_ok = True)

    for i in range(count):
        seed_idx = np.random.choice(len(note_dictionary), 1)[0]

        seed_tuple = note_dictionary[seed_idx]

        print(seed_tuple)

        output = generate_notes(
            model,
            seed_tuple,
            note_to_int, int_to_note,
            offset_to_int, int_to_offset,
            duration_to_int, int_to_duration,
            device,
            params["output_length"]
        )
        create_midi(output, os.path.join("./output", "output" + str(i) + ".mid"))

    #fs = FluidSynth("./soundfont.sf2")

    #fs.play_midi(os.path.join("./output", "output.mid"))

    #fs.midi_to_audio(os.path.join("./output", "output.mid"), os.path.join("./output", "output.mp3"))


    

        

if __name__ == "__main__":
    main(False)