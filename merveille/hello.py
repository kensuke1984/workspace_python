from pathlib import Path
from matplotlib import pyplot as plt

root = Path('D:/Documents/share/western_pacific/prem_spc/spcsac')


def get_events() -> list:
    import re
    return [i for i in root.glob('*') if re.search('\\d+[A-Z]', str(i))]


from typing import List


def get_transverse(event) -> List[str]:
    return [str(i) for i in event.glob('*.Ts')]


def cut(data):
    # start = 1000*20
    # end = 1200*20
    return data
    # pass


def down_sampl(data):
    return data[::20]


def data_label(event, label):
    sacfiles = get_transverse(event)
    return tuple((down_sampl(cut(read_sac(sf))), label) for sf in sacfiles)


if __name__ == '__main__':
    from readsac import read_sac
    import random
    import numpy as np

    events = get_events()
    sacfiles = get_transverse(events[0])
    tr = read_sac(sacfiles[0])
    # plt.plot(ds)
    data = np.empty((0, 2))
    j = 0
    # print(data)
    for i in events[:5]:
        daa = np.array(data_label(i, j))
        data = np.append(data, daa, axis=0)
        # data = data + np.array(list(data_label(i, j)))
        j = j + 1
    # data = np.array(data)

    random.shuffle(data)
    label=data[:,1]
    wave=data[:,0]
# print(read_sac(sacfiles[0]))
# plt.show()
