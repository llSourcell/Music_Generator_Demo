# author chinabluewu@github

import numpy as np
from tqdm import tqdm
import glob
#sudo pip install msgpack-python
import msgpack
import midi_manipulation

#files = glob.glob('{}/*.mid*'.format(path))
try:
    files = glob.glob('generated*.mid*')
except Exception as e:
    raise e

songs = np.zeros((0,156))
for f in tqdm(files):
    try:
        song = np.array(midi_manipulation.midiToNoteStateMatrix(f))

        if np.array(song).shape[0] > 10:
            #songs.append(song)
            songs = np.concatenate((songs,song))
    except Exception as e:
        raise e
print "samlpes merging ..."
print np.shape(songs)
midi_manipulation.noteStateMatrixToMidi(songs, "final")