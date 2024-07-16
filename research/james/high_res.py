#%%
#%%
#%%
from mlky import Config

c.analyze
c = Config('sudsaq/configs/high_res.yml', patch='default<-mac<-v1<-jan')
print(c.dumpYaml())

ls /Volumes/MLIA_active_data/data_SUDSAQ/data/momo/2020/01.nc
