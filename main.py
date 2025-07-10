from dataparser import notes, get_vocab
from model import LSTMModel, compute_loss
from trainer import train_step
from midi_parsing import write_midi
import torch
from tqdm import tqdm
from generator import generate_notes, generate_end
from torch.utils.data import DataLoader
from dataset import MIDISequenceDataset, OnTheFlyMIDIDataset
#from midi2audio import FluidSynth
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt




def main(trained = False) -> None:

    count = 5

    params = dict(
        embedding_dim = 512,
        hidden_size = 2048,
        batch_size = 16,
        sequence_length = 500,
        learning_rate = 5e-4,
        training_iterations = 10000,
        output_length = 200,
        max_end_length = 200,
        valid_offsets = [0, 1, 4, 7, 10, 15, 20, 24, 30, 40, 48, 60, 80, 96, 120, 160, 192, 240, 288, 360, 480, 600, 720, 960, 1200, 1440, 1920],
        valid_durations = [0, 8, 18, 26, 38, 47, 59, 75, 95, 113, 151, 181, 227, 275, 341, 455, 569, 683, 911, 1139, 1367, 1823, 2735, 3647, 5471],
    )

    offset_to_int = {u : i for i, u in enumerate(params["valid_offsets"])}
    duration_to_int = {u : i for i, u in enumerate(params["valid_durations"])}

    path = "midi_files"
    model_dir = "./models"

    model = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not trained:
        try: 
            with open(os.path.join(model_dir, "lists.pkl"), "rb") as f:
                lists = pickle.load(f)
                note_list, end_note_list, combined_note_list = lists["note_list"], lists["end_note_list"], lists["combined_note_list"]

        except:
            note_list, end_note_list, combined_note_list = notes(path, params["sequence_length"], params["max_end_length"])
        
        note_to_int, int_to_note = get_vocab(combined_note_list)

        note_dictionary = set()
        for midi_file in note_list:
            for note in midi_file:
                note_dictionary.add(note)

        note_dictionary = list(note_dictionary)

        max_offset = len(params["valid_offsets"])
        max_duration = len(params["valid_durations"])

        offset_to_int = {u : i for i, u in enumerate(params["valid_offsets"])}
        duration_to_int = {u : i for i, u in enumerate(params["valid_durations"])}

        # max_offset = max([max(note_file, key = lambda x: x[1])[1] for note_file in combined_note_list])
        # max_duration = max([max(note_file, key = lambda x: x[2])[2] for note_file in combined_note_list])
        max_volume = max([max(note_file, key = lambda x: x[3])[3] for note_file in combined_note_list])
        max_vals = [max_offset, max_duration, max_volume]

        vocabs = {
            "note_to_int": note_to_int,
            "int_to_note": int_to_note,
            "note_dictionary": note_dictionary,
            "maximum_vals": max_vals
        }

        with open(os.path.join(model_dir, "vocabs.pkl"), "wb") as f:
            pickle.dump(vocabs, f)
        vocab_size = len(int_to_note)


        """if not on_the_fly:
            input_sequences, output_sequences = generate_io_sequences(note_list, note_to_int, offset_to_int, duration_to_int, sequence_length = params["sequence_length"])
            dataset = MIDISequenceDataset(input_sequences, output_sequences)
        else: """

        

        dataset = OnTheFlyMIDIDataset(note_list, note_to_int, offset_to_int, duration_to_int, params["sequence_length"])
        dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True, drop_last=True)

        # I guess this decides what the training and model runs on
        model = LSTMModel(vocab_size, max_offset, max_duration, max_volume, params["embedding_dim"], params["hidden_size"]).to(device)
        

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
            x_pitch, x_offset, x_duration, x_volume = x_vals
            y_pitch, y_offset, y_duration, y_volume = y_vals
            x_pitch, x_offset, x_duration, x_volume = x_pitch.to(device), x_offset.to(device), x_duration.to(device), x_volume.to(device)
            y_pitch, y_offset, y_duration, y_volume = y_pitch.to(device), y_offset.to(device), y_duration.to(device), y_volume.to(device)

            # Stack inputs along last dimension for embedding
            x = torch.stack([x_pitch, x_offset, x_duration, x_volume], dim=2)  # shape: (batch, seq, 3)
            y = (y_pitch, y_offset, y_duration, y_volume)

            loss = train_step(x, y, model, optimizer, compute_loss, max_offset, max_duration, max_volume, device)
            losses.append(loss)
        torch.save(model.state_dict(), os.path.join(model_dir, "mdl.pth"))
        
        plt.plot(losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.savefig(os.path.join(model_dir, "loss_curve.png"))
        plt.clf()

        ending_model = LSTMModel(vocab_size, max_offset, max_duration, max_volume, params["embedding_dim"], params["hidden_size"]).to(device)
        ending_optimizer = torch.optim.Adam(ending_model.parameters(), lr=params["learning_rate"])
        if hasattr(tqdm, '_instances'): tqdm._instances.clear() # tqdm is the progress bar thing

        ending_dataset = OnTheFlyMIDIDataset(end_note_list, note_to_int, offset_to_int, duration_to_int, params["sequence_length"])
        ending_dataloader = DataLoader(ending_dataset, batch_size=params["batch_size"], shuffle=True, drop_last=True)

        ending_losses = []

        for it, (x_vals, y_vals) in enumerate(tqdm(ending_dataloader)):
            if it >= params["training_iterations"]:
                break
            elif it % 100 == 0:
                torch.save(ending_model.state_dict(), os.path.join(model_dir, "end_mdl.pth"))
            x_pitch, x_offset, x_duration, x_volume = x_vals
            y_pitch, y_offset, y_duration, y_volume = y_vals
            x_pitch, x_offset, x_duration, x_volume = x_pitch.to(device), x_offset.to(device), x_duration.to(device), x_volume.to(device)
            y_pitch, y_offset, y_duration, y_volume = y_pitch.to(device), y_offset.to(device), y_duration.to(device), y_volume.to(device)

            # Stack inputs along last dimension for embedding
            x = torch.stack([x_pitch, x_offset, x_duration, x_volume], dim=2)  # shape: (batch, seq, 3)
            y = (y_pitch, y_offset, y_duration, y_volume)

            end_loss = train_step(x, y, model, ending_optimizer, compute_loss)
            ending_losses.append(end_loss)
        torch.save(ending_model.state_dict(), os.path.join(model_dir, "end_mdl.pth"))

        plt.plot(ending_losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Ending Model Training Loss Curve")
        plt.savefig(os.path.join(model_dir, "end_loss_curve.png"))

    else:
        with open(os.path.join(model_dir, "vocabs.pkl"), "rb") as f:
            vocabs = pickle.load(f)
        note_to_int = vocabs["note_to_int"]
        int_to_note = vocabs["int_to_note"]
        note_dictionary = vocabs["note_dictionary"]
        max_offset, max_duration, max_volume = vocabs["maximum_vals"]

        model = LSTMModel(len(int_to_note), max_offset, max_duration, max_volume, params["embedding_dim"], params["hidden_size"]).to(device)

        model.load_state_dict(torch.load(os.path.join(model_dir, "mdl.pth"), map_location = device))
        model.eval()

        """ending_model = LSTMModel(
            vocab_size=len(int_to_note),
            num_offsets=len(int_to_offset),
            num_durations=len(int_to_duration),
            embedding_dim=params["embedding_dim"],
            hidden_size=params["hidden_size"]
        ).to(device)
        ending_model.load_state_dict(torch.load(os.path.join(model_dir, "end_mdl.pth"), map_location = device))"""
    os.makedirs("./output", exist_ok = True)

    for i in range(count):
        seed_idx = np.random.choice(len(note_dictionary), 1)[0]

        seed_tuple = note_dictionary[seed_idx]

        print(seed_tuple)

        output = generate_notes(
            model,
            seed_tuple,
            note_to_int, int_to_note,
            offset_to_int, params["valid_offsets"],
            duration_to_int, params["valid_durations"],
            device,
            params["output_length"]
        )

        print(output)

        """output = generate_end(
            model,
            output,
            note_to_int, int_to_note,
            offset_to_int, params["valid_offsets"],
            duration_to_int, params["valid_durations"],
            device,
            params["max_end_length"]
        )"""
        with open("text.txt", "w") as f:
            f.write(str(output))

        write_midi(output, os.path.join("./output", "output_v2_" + str(i) + ".mid"))

    #fs = FluidSynth("./soundfont.sf2")

    #fs.play_midi(os.path.join("./output", "output.mid"))

    #fs.midi_to_audio(os.path.join("./output", "output.mid"), os.path.join("./output", "output.mp3"))


    

        

if __name__ == "__main__":
    main(True)