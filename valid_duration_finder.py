from midi_parsing import parse_midi
import pickle
import os 

model_dir = "./models"

with open(os.path.join(model_dir, "lists.pkl"), "rb") as f:
    lists = pickle.load(f)
    note_list, end_note_list, combined_note_list = lists["note_list"], lists["end_note_list"], lists["combined_note_list"]

offset_occurence_dict = {}
duration_occurence_dict = {}

for file in combined_note_list:
    for _, offset, duration, _ in file:
        try: 
            offset_occurence_dict[offset] += 1
        except:
            offset_occurence_dict[offset] = 1
        
        try: 
            duration_occurence_dict[duration] += 1
        except:
            duration_occurence_dict[duration] = 1

offset_occurence_dict, duration_occurence_dict = dict(sorted(offset_occurence_dict.items(), key=lambda item: item[1], reverse = True)), dict(sorted(duration_occurence_dict.items(), key=lambda item: item[1], reverse = True))

standard_offset = 240
standard_duration = 227

valid_offsets = [0]
valid_durations = [0]

for i in range(-5, 5):
    if i != -5:
        valid_offsets.append(round(2 ** i * standard_offset / 3))
        valid_durations.append(round(2 ** i * standard_duration / 3))
    valid_offsets.append(round(2 ** i * standard_offset))
    valid_durations.append(round(2 ** i * standard_duration))
    valid_offsets.append(round(2 ** i * standard_offset * 3))
    valid_durations.append(round(2 ** i * standard_duration * 3))

offset_buckets = {}
duration_buckets = {}

for offset in offset_occurence_dict:
    if offset_occurence_dict[offset] < 200:
        break
    bucket = min(valid_offsets, key = lambda x: abs(x - offset))
    try: 
        offset_buckets[bucket] += 0
    except:
        offset_buckets[bucket] = offset

for duration in duration_occurence_dict:
    if duration_occurence_dict[duration] < 200:
        break
    bucket = min(valid_durations, key = lambda x: abs(x - duration))
    try: 
        duration_buckets[bucket] += 0
    except:
        duration_buckets[bucket] = duration

valid_offsets = sorted(offset_buckets.values())
valid_durations = sorted(duration_buckets.values())
print(valid_offsets)
print(valid_durations)


