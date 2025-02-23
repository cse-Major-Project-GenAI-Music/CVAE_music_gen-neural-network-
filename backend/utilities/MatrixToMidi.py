from mido import Message, MidiFile, MidiTrack
import torch 

def generate_midi_from_matrix(sample, output_path):
    """Generate a MIDI file from the given sample (500x128)."""
    instrument_map = {1: 0, 2: 25, 3: 33}  # Map values to MIDI instruments
    midi_file = MidiFile()
    tracks = {instrument: MidiTrack() for instrument in instrument_map.values()}
    
    for instrument, track in tracks.items():
        midi_file.tracks.append(track)
        track.append(Message('program_change', program=instrument, time=0))
    
    active_notes = {instrument: {} for instrument in instrument_map.values()}  # {instrument: {note: start_time}}
    
    sample = torch.tensor(sample)
    sample = sample.int()
    
    for t in range(sample.shape[0]):  # Iterate over 500 timesteps
        abs_time = t * 100  # Absolute time reference in MIDI ticks
        
        for note in range(sample.shape[1]):  # Iterate over 128 notes
            value = int(sample[t, note])  # Explicitly cast to int
            instrument = instrument_map.get(value, None)
            
            if instrument is not None:  # Note is played by an instrument
                track = tracks[instrument]
                if note in active_notes[instrument]:
                    prev_time = active_notes[instrument][note]
                    if t - prev_time <= 4:
                        continue  # Merge event if close enough
                     
                    else:
                        track.append(Message('note_off', note=note, velocity=127, time=(t - prev_time) * 100))
                        del active_notes[instrument][note]
                
                if note not in active_notes[instrument]:
                    track.append(Message('note_on', note=note, velocity=127, time=0))
                    active_notes[instrument][note] = t
            
            for inst, notes in active_notes.items():
                if inst != instrument and note in notes:
                    tracks[inst].append(Message('note_off', note=note, velocity=127, time=(t - notes[note]) * 100))
                    del notes[note]
    
    # Ensure all active notes are turned off before saving
    for instrument, notes in active_notes.items():
        track = tracks[instrument]
        for note, start_time in notes.items():
            track.append(Message('note_off', note=note, velocity=127, time=(500 - start_time) * 100))
    
    midi_file.save(output_path)
    print(f"MIDI file saved at {output_path}")