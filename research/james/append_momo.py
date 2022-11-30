#%%
#%%
#%%
import os
import xarray as xr

ns = xr.open_mfdataset('raw/*.nc')
ns = ns.rename({key: f'momo.{key}' for key in ns})

for year, yds in ns.groupby('time.year'):
    for month, mds in yds.groupby('time.month'):
        month = f'{month:02}'
        print(f'Processing {year}/{month}')
        if not os.path.exists(f'{year}/{month}.nc'):
            print('No momo file for this year/month')
            continue
        momo = xr.open_dataset(f'{year}/{month}.nc')
        ds   = xr.merge([momo, mds])
        # Remove 2dsfc
        ds   = ds.rename({key: f"momo.{key.split('.')[-1]}" for key in ds if '2dsfc' in key})
        if not (ds.time == momo.time).all():
            print('Merge did not produce the same time dimension, skipping')
        else:
            if not os.path.exists(f'updated/{year}'):
                os.mkdir(f'updated/{year}', mode=0o771)
            ds.to_netcdf(f'updated/{year}/{month}.nc')

#%%

keys = ['momo.2dsfc.Br2', 'momo.2dsfc.BrCl', 'momo.2dsfc.BrONO2', 'momo.2dsfc.BrOX', 'momo.2dsfc.C10H16', 'momo.2dsfc.C2H5OOH', 'momo.2dsfc.C2H6', 'momo.2dsfc.C3H6', 'momo.2dsfc.C3H7OOH', 'momo.2dsfc.C5H8', 'momo.2dsfc.CCl4', 'momo.2dsfc.CFC11', 'momo.2dsfc.CFC113', 'momo.2dsfc.CFC12', 'momo.2dsfc.CH2O', 'momo.2dsfc.CH3Br', 'momo.2dsfc.CH3CCl3', 'momo.2dsfc.CH3CHO', 'momo.2dsfc.CH3COCH3', 'momo.2dsfc.CH3COO2', 'momo.2dsfc.CH3COOOH', 'momo.2dsfc.CH3Cl', 'momo.2dsfc.CH3O2', 'momo.2dsfc.CH3OH', 'momo.2dsfc.CH3OOH', 'momo.2dsfc.CHBr3', 'momo.2dsfc.Cl2', 'momo.2dsfc.ClONO2', 'momo.2dsfc.ClOX', 'momo.2dsfc.DCDT.HOX', 'momo.2dsfc.DCDT.OY', 'momo.2dsfc.DCDT.SO2', 'momo.2dsfc.DMS', 'momo.2dsfc.H1211', 'momo.2dsfc.H1301', 'momo.2dsfc.H2O2', 'momo.2dsfc.HACET', 'momo.2dsfc.HBr', 'momo.2dsfc.HCFC22', 'momo.2dsfc.HCl', 'momo.2dsfc.HNO3', 'momo.2dsfc.HNO4', 'momo.2dsfc.HO2', 'momo.2dsfc.HOBr', 'momo.2dsfc.HOCl', 'momo.2dsfc.HOROOH', 'momo.2dsfc.ISON', 'momo.2dsfc.ISOOH', 'momo.2dsfc.LR.HOX', 'momo.2dsfc.LR.OY', 'momo.2dsfc.LR.SO2', 'momo.2dsfc.MACR', 'momo.2dsfc.MACROOH', 'momo.2dsfc.MGLY', 'momo.2dsfc.MPAN', 'momo.2dsfc.N2O5', 'momo.2dsfc.NALD', 'momo.2dsfc.NH3', 'momo.2dsfc.NH4', 'momo.2dsfc.OCS', 'momo.2dsfc.OH', 'momo.2dsfc.ONMV', 'momo.2dsfc.PAN', 'momo.2dsfc.PROD.HOX', 'momo.2dsfc.PROD.OY', 'momo.2dsfc.SO2', 'momo.2dsfc.SO4', 'momo.2dsfc.dflx.bc', 'momo.2dsfc.dflx.dust', 'momo.2dsfc.dflx.hno3', 'momo.2dsfc.dflx.nh3', 'momo.2dsfc.dflx.nh4', 'momo.2dsfc.dflx.oc', 'momo.2dsfc.dflx.salt', 'momo.2dsfc.dms', 'momo.2dsfc.doxdyn', 'momo.2dsfc.doxphy', 'momo.2dsfc.mc.bc', 'momo.2dsfc.mc.dust', 'momo.2dsfc.mc.nh4', 'momo.2dsfc.mc.nitr', 'momo.2dsfc.mc.oc', 'momo.2dsfc.mc.pm25.dust', 'momo.2dsfc.mc.pm25.salt', 'momo.2dsfc.mc.salt', 'momo.2dsfc.mc.sulf', 'momo.2dsfc.taut', 'momo.T2', 'momo.ccover', 'momo.ccoverh', 'momo.ccoverl', 'momo.ccoverm', 'momo.cumf', 'momo.cumf0', 'momo.dqcum', 'momo.dqdad', 'momo.dqdyn', 'momo.dqlsc', 'momo.dqvdf', 'momo.dtcum', 'momo.dtdad', 'momo.dtdyn', 'momo.dtlsc', 'momo.dtradl', 'momo.dtrads', 'momo.dtvdf', 'momo.evap', 'momo.olr', 'momo.olrc', 'momo.osr', 'momo.osrc', 'momo.prcp', 'momo.prcpc', 'momo.prcpl', 'momo.precw', 'momo.q2', 'momo.sens', 'momo.slrc', 'momo.slrdc', 'momo.snow', 'momo.ssrc', 'momo.taugxs', 'momo.taugys', 'momo.taux', 'momo.tauy', 'momo.twpc', 'momo.u10', 'momo.uvabs', 'momo.v10', 'momo.aerosol.nh4', 'momo.aerosol.no3', 'momo.aerosol.sul', 'momo.ch2o', 'momo.co', 'momo.hno3', 'momo.oh', 'momo.pan', 'momo.ps', 'momo.q', 'momo.so2', 'momo.t', 'momo.u', 'momo.v', 'momo.mda8', 'momo.o3']


rename =

len(rename)


len(keys)

rename

#%%
import xarray as xr

ds = xr.open_dataset('local/data/MOMO-extra_201309.nc')

ds = ds[['u10', 'v10']]

ds.load()
ds
#%%

ds.chunk()

import numpy as np

np.split(ds, 10)

ds

data = ds.to_array().stack({'loc': ['lat', 'lon', 'time']})
data = data.transpose('loc', 'variable')
# np.split(data, 10, axis=1)
#%%
from tqdm import tqdm

# copy = xr.zeros_like(data)
copies = []
for split in tqdm(np.split(data, 10), desc='Processed Splits'):
    copy = xr.zeros_like(split)
    copy[:] = split[:]
    copies.append(copy)
    # copy.loc[{'loc': split['loc']}] = split

#%%
copy = xr.concat(copies, 'loc')

copy.identical(data)
copy

(copy == data).all()

#%%

data

split

#%%

ns = ds.rename({'u10': 'momo.u10', 'v10': 'toar.v10'})

from sudsaq.data import Dataset

ns = Dataset(ns)

ns['new'] = xr.zeros_like(ns['momo.u10'])

ns['(?!toar).*']

#%%
copy

copy.sel(flat=split.flat)

split




copy.loc[{'flat': split.flat}]


#%%

data.shape[1] // 10

#%%

def process(chunk):
    print(chunk)
    return chunk

chunked = data.chunk(flat=10)
chunked.map_blocks(process)

#%%

files = """\
2to3 -> 2to3-3.10
bzcmp -> bzdiff
bzegrep -> bzgrep
bzfgrep -> bzgrep
bzless -> bzmore
captoinfo -> tic
idle3 -> idle3.10
infotocap -> tic
lz4c -> lz4
lz4cat -> lz4
lzcat -> xz
lzcmp -> xzdiff
lzdiff -> xzdiff
lzegrep -> xzgrep
lzfgrep -> xzgrep
lzgrep -> xzgrep
lzless -> xzless
lzma -> xz
lzmore -> xzmore
pydoc -> pydoc3.10
pydoc3 -> pydoc3.10
python -> python3.10
python3 -> python3.10
python3-config -> python3.10-config
python3.1 -> python3.10
reset -> tset
tclsh -> tclsh8.6
unlz4 -> lz4
unlzma -> xz
unxz -> xz
unzstd -> zstd
wish -> wish8.6
xzcat -> xz
xzcmp -> xzdiff
xzegrep -> xzgrep
xzfgrep -> xzgrep
zstdcat -> zstd
zstdmt -> zstd\
"""
import shutil

files = files.split('\n')
for file in files:
    copy, orig = file.split(' -> ')
    shutil.copy(orig, copy)


#%%

path = '/Users/jamesmo/projects/dockers/interactive/mounts/mlia-active-data/data_SUDSAQ/conda/gattaca/'
path = pathlib.Path(path)

i = 0
for file in path.rglob('*'):
    if file.is_symlink():
        i += 1
        print(file)
print(f'Total files: {i}')


#%%
import pathlib

path = '/Users/jamesmo/projects/dockers/interactive/mounts/mlia-active-data/data_SUDSAQ/conda/gattaca/'
path = pathlib.Path(path)

i, e = 0, 0
for file in path.rglob('*'):
    if file.is_symlink():
        try:
            orig = f'{file.parent}/{file.readlink()}'
            file.unlink()
            shutil.copy(orig, file)
            i += 1
        except:
            e += 1

print(f'Processed {i}/{i+e} files')

#%%
file.unlink()
file

help(file.unlink)

file.readlink()

file

#%%
f = pathlib.Path('/Users/jamesmo/projects/dockers/interactive/mounts/mlia-active-data/data_SUDSAQ/conda/gattaca/bin/python3')

f.is_symlink()
f.readlink()

f'{f.parent}/{f.readlink()}'

#%%

string = 'v10 + 2v10'

for key in list(ds):
    if key in string:
        string = f"ds['{key}']".join(string.split(key))

#%%


def replace(string):
    for key in keys:
        string = re.sub(fr'({key})(?!\d|\w)', f"ds['{key}']", string)

    return string

keys = ['momo.no', 'momo.no2', 'momo.t']
replace('momo.no2**2')
replace('momo.no/(momo.no2*momo.no)')


data['loc'].size

ds

#%%
# (?!toar|momo\.o3).*
ns['(?!toar|momo.o3|new).*']
ns['momo.o3'] = ns['new']
ns

#%%

np.isfinite

ns['new'][0, 0, 0] = np.inf
ns.where(~np.isfinite(ns))
ns[ns == np.inf]


help(ns.where)
