import mido
from math import log

def standardize_duration(d: int, is_duration: bool):
    # is_duration determines whether it is an offset or a duration. 
    if d == 0:
        return 0
    valid_offsets = [1, 4, 7, 10, 15, 20, 24, 30, 40, 48, 60, 80, 96, 120, 160, 192, 240, 288, 360, 480, 600, 720, 960, 1200, 1440, 1920]
    valid_durations = [8, 18, 26, 38, 47, 59, 75, 95, 113, 151, 181, 227, 275, 341, 455, 569, 683, 911, 1139, 1367, 1823, 2735, 3647, 5471]

    if is_duration:
        return min(valid_durations, key = lambda x: abs(log(x) - log(d)))
    else: 
        return min(valid_offsets, key = lambda x: abs(log(x) - log(d)))


def parse_midi(filepath, piano_roll = True):
    mid = mido.MidiFile(filepath)
    notes = []
    abs_time = 0
    note_on_dict = {}

    # print(len(mid.tracks))

    for track in mid.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            # Note on
            if msg.type == 'note_on' and msg.velocity > 0:
                note_on_dict[(msg.channel, msg.note)] = (abs_time, msg.velocity)
            # Note off (or note_on with velocity 0)
            elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in note_on_dict:
                    start, velocity = note_on_dict.pop(key)
                    duration = abs_time - start
                    notes.append((msg.note, start, duration, velocity))
            # Pedal (sustain)
            elif msg.type == 'control_change' and msg.control == 64:
                notes.append(("pedal_down" if msg.value >= 64 else "pedal_up", abs_time, 0, 0))


    notes.sort(key = lambda x: x[1])

    duplicate_pedal_removed_notes = []
    last_pedal = 0
    for i in range(len(notes)):
        if last_pedal == "pedal_down" and notes[i][0] == "pedal_down":
            continue
        elif last_pedal == "pedal_up" and notes[i][0] == "pedal_up":
            continue
        duplicate_pedal_removed_notes.append(notes[i])
        if type(notes[i][0]) != type(1):
            last_pedal = notes[i][0]

    notes = duplicate_pedal_removed_notes

    for i in range(len(notes) - 2, -1, -1):
        notes[i + 1] = (notes[i + 1][0], standardize_duration(notes[i + 1][1] - notes[i][1], False), standardize_duration(notes[i + 1][2], True), notes[i + 1][3])

    notes[0] = (notes[0][0], standardize_duration(notes[0][1], False), standardize_duration(notes[0][2], True), notes[0][3])

    if piano_roll:
        return notes

    offset_bunched_notes = []
    last_offset_shorted_note_set = set()
    for note in notes:
        if note[1] == 0:
            last_offset_shorted_note_set.add((note[0], note[2], note[3]))
        else: 
            offset_bunched_notes.append((note[1], last_offset_shorted_note_set))
            last_offset_shorted_note_set.clear()

    return offset_bunched_notes

def write_midi(notes, output_path, tempo=500000):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    abs_time = 0

    # Set tempo
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

    # Combine notes and pedals, sort by start time
    events = []
    for note, offset, duration, velocity in notes:
        # case pedal
        if note == "pedal_up":
            events.append((abs_time + offset, 'control_change', 64, 0))
        elif note == "pedal_down":
            events.append((abs_time + offset, 'control_change', 64, 127))
        else:
            events.append((abs_time + offset, 'note_on', note, velocity))
            events.append((abs_time + offset + duration, 'note_off', note, velocity))
        abs_time += offset
    events.sort(key = lambda x: x[0])

    last_time = 0
    for event in events:
        delta = int(event[0] - last_time)
        last_time = event[0]
        if event[1] == 'note_on':
            track.append(mido.Message('note_on', note=event[2], velocity=event[3], time=delta))
        elif event[1] == 'note_off':
            track.append(mido.Message('note_off', note=event[2], velocity=event[3], time=delta))
        elif event[1] == 'control_change':
            track.append(mido.Message('control_change', control=event[2], value=event[3], time=delta))

    mid.save(output_path)