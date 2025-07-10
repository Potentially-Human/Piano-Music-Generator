from dataparser import notes
import pickle
import os

model_dir = "./models"
path = "midi_files"

note_list, end_note_list, combined_note_list = notes(path, 300, 50)

lists = {
    "note_list": note_list,
    "end_note_list": end_note_list,
    "combined_note_list": combined_note_list,
}

with open(os.path.join(model_dir, "lists.pkl"), "wb") as f:
    pickle.dump(lists, f)