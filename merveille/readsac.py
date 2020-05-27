import numpy as np


def read_sac(filename):
    from util import SACFileName
    from obspy import read
    if isinstance(filename, SACFileName):
        tr = read(str(filename.get_path()))[0]
    else:
        tr = read(str(filename))[0]
    return tr.data


def normalize(data: np.ndarray):
    return data / np.max(np.abs(data))


if __name__ == '__main__':
    a = read_sac("D:/Documents/share/western_pacific/prem_spc/spcsac/010202E/ABU.010202E.Ts")
    print(type(a))
