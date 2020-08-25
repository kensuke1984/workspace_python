from functools import reduce
from pathlib import Path

import obspy
from obspy import Trace, Stream, read_inventory

import gcmt


class SACFileName:

    def __init__(self, path: Path):
        """

        :param path: of SACFile created by mseed2sac, must be like '#YS.PHRM.00.BHZ.M.2004.349.231019.SAC'
        """
        self.__path = path
        self.name = path.name
        self.network = path.name.split('.')[0]
        self.station = path.name.split('.')[1]
        self.loc = path.name.split('.')[2]
        self.channel = path.name.split('.')[3]
        self.unknown_m = path.name.split('.')[4]  # TODO
        self.year = path.name.split('.')[5]
        self.day_of_year = path.name.split('.')[6]
        self.time = path.name.split('.')[7]

    def get_path(self):
        return self.__path

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Meta:
    def __init__(self, net, sta, loc, channel, lat, lon, elev, depth, azimuth, dip, instrument, scale, scalefreq,
                 scaleunits, samplerate, start, end):
        self.net = net
        self.sta = sta
        self.loc = loc
        self.channel = channel
        self.lat = float(lat)
        self.lon = float(lon)
        self.elev = float(elev)
        self.depth = float(depth)
        self.azimuth = float(azimuth)
        self.dip = float(dip)
        self.instrument = instrument
        self.scale = scale
        self.scalefreq = scalefreq
        self.scaleunits = scaleunits
        self.samplerate = samplerate
        self.start = start
        self.end = end

    def get_fileglob(self):
        return '.'.join([self.net, self.sta, '' if self.loc == '--' else self.loc, self.channel,
                         '*'])


class MetaFile:
    def __init__(self, path: Path):
        metas = []
        self.path = path
        with open(path) as f:
            for line in f.readlines():
                if line.startswith('#'):
                    continue
                metas.append(Meta(*line.split('|')))
        self.metas = metas

    def get_metafile(self):
        return self.path.parent.joinpath(self.path.name.replace('.meta', '.xml'))


def add_metainfo(tr: Trace, meta: Meta):
    header = tr.stats.sac
    header['stla'] = meta.lat
    header['stlo'] = meta.lon
    return tr


def remove_resp(tr: Trace, xml_path: Path):
    from obspy import read_inventory
    inv = read_inventory(str(xml_path))
    pre_filt = [0.001, 0.005, 10, 20]
    tr.remove_response(inventory=inv, pre_filt=pre_filt, output="DISP",
                       water_level=60, plot=False)
    return tr


def extract(sacpath: Path, xmlpath: Path):
    from obspy import read
    st = read(str(sacpath))
    return remove_resp(st[0], xmlpath)


def time_adjust(tr: Trace):
    from datetime import timedelta
    centroid_time = obspy.UTCDateTime(event_info.centroid_time)
    header = tr.stats.sac
    tr.trim(centroid_time, centroid_time + timedelta(seconds=tlen))
    header['nzjday'] = centroid_time.julday
    header['nzhour'] = centroid_time.hour
    header['nzmin'] = centroid_time.minute
    header['nzsec'] = centroid_time.second
    header['nzmsec'] = centroid_time.microsecond
    return tr


def rotate(st: Stream, xml_path: Path):
    inv = read_inventory(str(xml_path))
    st.rotate('NE->RT', inventory=inv, back_azimuth=st[1].stats.sac['baz'])
    # st.write('/tmp/hoge.SAC')
    station = st[0].get_id()
    print(station)
    st[2].plot()
    # st.plot()
    return st


def make_stream(z_tr: Trace, n_tr: Trace, e_tr: Trace):
    return Stream(traces=[z_tr, n_tr, e_tr])


def merge(files):
    from numpy import ndarray
    files = [str(f) for f in files]
    print(files)
    sts = [obspy.read(f) for f in files]
    st = reduce(lambda a, b: a + b, sts)
    st.sort(['starttime'])
    st.merge(method=0)
    if not isinstance(st[0].data, ndarray):
        st[0].data = st[0].data.filled(0)
    return st[0]


def to_ne_path(ne, zpath: Path):
    bne = '.BHN.' if ne == 'N' else '.BHE.'
    hne = '.HHN.' if ne == 'N' else '.HHE.'
    return zpath.with_name(str(zpath.name).replace('.BHZ.', bne).replace('.HHZ.', hne))


def rotate_process():
    print('jiq')
    for z_path in output_path.glob('*.?HZ.*.SAC'):
        n_path = to_ne_path('N', z_path)
        e_path = to_ne_path('E', z_path)
        if not n_path.exists() or not e_path.exists():
            continue
        # zsac = obspy.read(str(z_path))
        # nsac = obspy.read(str(n_path))
        # esac = obspy.read(str(e_path))
        # znesac = make_stream(zsac[0], nsac[0], esac[0])
        # rotate(znesac)
        # print(z_path)


exit()

# print(zensac[0].stats.sac['baz'])
# rotate(zensac, Path('/Users/kensuke/workspace/anselme/_mseed2sac/IRISDMC-._HAWAII.xml'))
pass

# ------------------------------------
sacfiles_path = Path('/Users/kensuke/workspace/anselme/_mseed2sac')
metafiles_path = Path('/Users/kensuke/workspace/anselme/meta')
output_path = Path('/Users/kensuke/workspace/anselme/output')
cmtid = '121404B'
tlen = 3276.8
# -------------------------------------


event_info = gcmt.of(cmtid)
metafiles = list(metafiles_path.glob('*.meta'))
rotate_process()
exit()
for metafile in metafiles:
    metas = MetaFile(metafile).metas
    xmlfile = metafile.with_suffix('.xml')
    for meta in metas[0:5]:
        sacfile = list(sacfiles_path.glob(meta.get_fileglob()))
        if len(sacfile) == 0:
            continue
        tr = merge(sacfile)
        tr = add_metainfo(tr, meta)
        tr = event_info.addinfo_to(tr)
        tr = remove_resp(tr, xmlfile)
        tr = time_adjust(tr)
        name = '.'.join([meta.net, meta.sta, '' if meta.loc == '--' else meta.loc, meta.channel,
                         str(tr.stats.starttime.year), str(tr.stats.starttime.julday), 'SAC'])
        tr.write(str(output_path.joinpath(name)))
        print(name)
        # exit()
