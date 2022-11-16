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
