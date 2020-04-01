def read_sac(filename):
    from obspy import read
    tr = read(filename)[0]
    return tr.data


if __name__ == '__main__':
    a = read_sac("D:/Documents/share/western_pacific/prem_spc/spcsac/010202E/ABU.010202E.Ts")
    print(type(a))
