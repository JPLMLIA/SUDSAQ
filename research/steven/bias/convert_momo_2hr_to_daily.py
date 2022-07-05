import calendar
import numpy as np
import xarray as xr
import datetime as dt


def main(in_momo_nc, out_momo_nc, year, month):
    ns = xr.open_mfdataset(in_momo_nc, engine='scipy', parallel=True)
    ns.load()

    mda8 = ns['mda8'][0, :, :]

    time = ns.time.dt.time  # Convert the times dimension to datetime.time objects
    mask = (dt.time(8) < time) & (time < dt.time(16)) | (time == dt.time(0))
    # Select timestamps between 8am and 4pm OR midnight

    ss = ns.where(mask, drop=True)  # Drop timestamps that don't match
    gs = ss.groupby('time.day').mean()  # Take the mean by day
    gs = gs.rename(day='time')
    n_days = calendar.monthrange(year, month)[1]
    # gs['time'] = np.array([dt.date(year, month, day).strftime('%Y-%m-%d')
    #                        for day in range(1, n_days + 1)])
    gs['time'] = np.array([dt.datetime(year, month, day)
                           for day in range(1, n_days + 1)])
    gs['mda8'] = mda8.astype(np.float64)

    gs.to_netcdf(out_momo_nc, mode='w', engine='scipy')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('in_momo_nc', type=str)
    parser.add_argument('out_momo_nc', type=str)
    parser.add_argument('year', type=int)
    parser.add_argument('month', type=int)

    args = parser.parse_args()
    main(**vars(args))
