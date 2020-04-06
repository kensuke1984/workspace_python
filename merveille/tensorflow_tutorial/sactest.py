import obspy

from obspy import read

st = read('D:\Documents/share/western_pacific/prem_spc/spcsac/011104C/ABU.011104C.Ts')  # load example seismogram
st.filter(type='highpass', freq=3.0)
# st = st.select(component='T')
data= st[0]
print(data.data)
