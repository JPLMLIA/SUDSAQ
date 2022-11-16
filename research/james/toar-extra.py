#%%
%load_ext autoreload
%autoreload 2
%matplotlib inline
#%%

import xarray as xr

#%%
ls /Volumes/MLIA_active_data/data_SUDSAQ/data/toar
ds = xr.open_dataset('/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/matched/2012/01.nc')

#%%

ds

md = xr.open_dataset('local/data/dev/toar/metadata.test.nc')
md

#%%

files = [
    '/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/matched/2012/01.nc',
    '/Users/jamesmo/projects/suds-air-quality/local/data/dev/toar/metadata.test.nc'
]

ds = xr.open_mfdataset(files, parallel=True)

ds.load()

#%%
data = ds.stack({'loc': ['lat', 'lon', 'time']})
data

#%%
data['toar.station_alt.std'].mean()
md['toar.station_alt.std'].mean()

data.isel(loc=50600)

md.isel(lat=50, lon=50)

md['toar.station_alt.mean'].where(~md['toar.station_alt.mean'].isnull())

#%%

ds.sel(lon=[])

#%%

tl = ds.drop_dims('time')

#%%

-140+180, -50+180

convert = lambda lon1, lon2: (lon1 + 360 if lon1 < 0 else lon1, lon2 + 360 if lon2 < 0 else lon2)
convert(-140, -50)
convert(-5, 5)

220 < lat < 310
355 < lat < 5

ds

#%%

ns = xr.open_mfdataset('/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/2005/01.nc')
ns
#%%

ns.sel(lon=5)
ns.where(ns.lon<5, ns.lon>355, drop=True)

ns.sel({'lon': slice(355, 360), 'lon': slice(0, 5)})

#%%
# ls /Volumes/MLIA_active_data/data_SUDSAQ/data/toar/matched/


#%%

data

ds
ns


sum([d.lon.size for d in data])

#%%

s = set(ds.lon.values)


s - set(ns.lon.values)

#%%


sub.time +

sub.time




help(np.timedelta64)


#%%

ns.drop_dims('time').any()

ps = ns.drop_dims('time')

len(ns)
len(ps)
ns

list(ns)
len(ps)
ps


a = ns.to_array()

a['variable']


#%%

for v in a['variable']:
    d = a.sel(variable=v)

d

#%%

>>> path  = 'Z:/data_SUDSAQ/model/new/model_data\\*\\combined\\test.data.nc'
>>> split = path.split('\\')
>>> split
['Z:/data_SUDSAQ/model/new/model_data', '*', 'combined', 'test.data.nc']
>>> split[1]
'*'

"""
2022-09-02 12:53:00,741 WARNING worker.py:1829 -- The node with node id: 5ad7fa5dd64fc1fec7a82cceecabed8e5eb247ec97064bb50a1c2a11 and address: 137.78.248.21 and node name: 137.78.248.21 has been marked dead because the detector has missed too many heartbeats from it. This can happen when a    (1) raylet crashes unexpectedly (OOM, preempted node, etc.)
        (2) raylet has lagging heartbeats due to slow network or busy workload.
./monthly.sh: line 32: 3376060 Killed                  nice -n 19 python ml/create.py -c $config -s $section
"""

#%%

str(ds['variable'])

list(ds)

#%%
import xarray as xr

files = [
    '/Volumes/MLIA_active_data/data_SUDSAQ/data/toar/matched/2012/01.nc',
    '/Volumes/MLIA_active_data/data_SUDSAQ/data/momo/2012/01.nc'
]

ds = xr.open_mfdataset(files, parallel=True)

#%%
from sudsaq.data import Dataset

ds = Dataset(ds)



#%%

ex = ['momo.hno3', 'momo.oh', 'momo.pan', 'momo.q2',
           'momo.sens', 'momo.so2', 'momo.T2', 'momo.taugxs',
           'momo.taugys', 'momo.taux', 'momo.tauy', 'momo.twpc',
           'momo.2dsfc.CFC11', 'momo.2dsfc.CFC113', 'momo.2dsfc.CFC12',
           'momo.ch2o', 'momo.cumf0', 'momo.2dsfc.dms'
]

len(ex)
'|'.join([key[5:] for key in ex])


#%%

r = '(momo|toar).(?!hno3|oh|pan|q2|sens|so2|T2|taugxs|taugys|taux|tauy|twpc|2dsfc.CFC11|2dsfc.CFC113|2dsfc.CFC12|ch2o|cumf0|2dsfc.dms).*'
ds[r]

#%%
ns = ds[['momo.v', 'momo.t', 'momo.u']]

ns.load()

#%%

from sklearn.preprocessing import StandardScaler

ns

scaler = StandardScaler()

scaler(ns)
scaler.fit(ns)

#%%

scaler  = StandardScaler(**config.input.StandardScaler)
data[:] = StandardScaler.fit_transform(data)


#%%

l = ['201201_2dsfc_Br2.nc', '201201_2dsfc_BrCl.nc', '201201_2dsfc_BrONO2.nc', '201201_2dsfc_BrOX.nc', '201201_2dsfc_C10H16.nc', '201201_2dsfc_C2H5OOH.nc', '201201_2dsfc_C2H6.nc', '201201_2dsfc_C3H6.nc', '201201_2dsfc_C3H7OOH.nc', '201201_2dsfc_C5H8.nc', '201201_2dsfc_CCl4.nc', '201201_2dsfc_CFC11.nc', '201201_2dsfc_CFC113.nc', '201201_2dsfc_CFC12.nc', '201201_2dsfc_CH2O.nc', '201201_2dsfc_CH3Br.nc', '201201_2dsfc_CH3CCl3.nc', '201201_2dsfc_CH3CHO.nc', '201201_2dsfc_CH3COCH3.nc', '201201_2dsfc_CH3COO2.nc', '201201_2dsfc_CH3COOOH.nc', '201201_2dsfc_CH3Cl.nc', '201201_2dsfc_CH3O2.nc', '201201_2dsfc_CH3OH.nc', '201201_2dsfc_CH3OOH.nc', '201201_2dsfc_CHBr3.nc', '201201_2dsfc_Cl2.nc', '201201_2dsfc_ClONO2.nc', '201201_2dsfc_ClOX.nc', '201201_2dsfc_DCDT_HOX.nc', '201201_2dsfc_DCDT_OY.nc', '201201_2dsfc_DCDT_SO2.nc', '201201_2dsfc_DMS.nc', '201201_2dsfc_H1211.nc', '201201_2dsfc_H1301.nc', '201201_2dsfc_H2O2.nc', '201201_2dsfc_HACET.nc', '201201_2dsfc_HBr.nc', '201201_2dsfc_HCFC22.nc', '201201_2dsfc_HCl.nc', '201201_2dsfc_HNO3.nc', '201201_2dsfc_HNO4.nc', '201201_2dsfc_HO2.nc', '201201_2dsfc_HOBr.nc', '201201_2dsfc_HOCl.nc', '201201_2dsfc_HOROOH.nc', '201201_2dsfc_ISON.nc', '201201_2dsfc_ISOOH.nc', '201201_2dsfc_LR_HOX.nc', '201201_2dsfc_LR_OY.nc', '201201_2dsfc_LR_SO2.nc', '201201_2dsfc_MACR.nc', '201201_2dsfc_MACROOH.nc', '201201_2dsfc_MGLY.nc', '201201_2dsfc_MPAN.nc', '201201_2dsfc_N2O5.nc', '201201_2dsfc_NALD.nc', '201201_2dsfc_NH3.nc', '201201_2dsfc_NH4.nc', '201201_2dsfc_OCS.nc', '201201_2dsfc_OH.nc', '201201_2dsfc_ONMV.nc', '201201_2dsfc_PAN.nc', '201201_2dsfc_PROD_HOX.nc', '201201_2dsfc_PROD_OY.nc', '201201_2dsfc_SO2.nc', '201201_2dsfc_SO4.nc', '201201_2dsfc_dflx_bc.nc', '201201_2dsfc_dflx_dust.nc', '201201_2dsfc_dflx_hno3.nc', '201201_2dsfc_dflx_nh3.nc', '201201_2dsfc_dflx_nh4.nc', '201201_2dsfc_dflx_oc.nc', '201201_2dsfc_dflx_salt.nc', '201201_2dsfc_dms.nc', '201201_2dsfc_doxdyn.nc', '201201_2dsfc_doxphy.nc', '201201_2dsfc_mc_bc.nc', '201201_2dsfc_mc_dust.nc', '201201_2dsfc_mc_nh4.nc', '201201_2dsfc_mc_nitr.nc', '201201_2dsfc_mc_oc.nc', '201201_2dsfc_mc_pm25_dust.nc', '201201_2dsfc_mc_pm25_salt.nc', '201201_2dsfc_mc_salt.nc', '201201_2dsfc_mc_sulf.nc', '201201_2dsfc_taut.nc', '201201_T2.nc', '201201_ccover.nc', '201201_ccoverh.nc', '201201_ccoverl.nc', '201201_ccoverm.nc', '201201_cumf.nc', '201201_cumf0.nc', '201201_dqcum.nc', '201201_dqdad.nc', '201201_dqdyn.nc', '201201_dqlsc.nc', '201201_dqvdf.nc', '201201_dtcum.nc', '201201_dtdad.nc', '201201_dtdyn.nc', '201201_dtlsc.nc', '201201_dtradl.nc', '201201_dtrads.nc', '201201_dtvdf.nc', '201201_evap.nc', '201201_olr.nc', '201201_olrc.nc', '201201_osr.nc', '201201_osrc.nc', '201201_prcp.nc', '201201_prcpc.nc', '201201_prcpl.nc', '201201_precw.nc', '201201_q2.nc', '201201_sens.nc', '201201_slrc.nc', '201201_slrdc.nc', '201201_snow.nc', '201201_ssrc.nc', '201201_taugxs.nc', '201201_taugys.nc', '201201_taux.nc', '201201_tauy.nc', '201201_twpc.nc', '201201_u10.nc', '201201_uvabs.nc', '201201_v10.nc']
len(l)
l
