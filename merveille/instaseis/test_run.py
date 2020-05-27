import instaseis

db  = instaseis.open_db('/home/kensuke/workspace/share/10s_PREM_ANI_FORCES')
db  = instaseis.open_db('http://ds.iris.edu/files/syngine/axisem/models/prem_a_5s_merge_compress2/merged_output.nc4')
print(db)