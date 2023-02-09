import pandas as pd
import requests

from tqdm import tqdm

def query(path, flags, pre='', verbose=True):
    url = 'https://toar-data.fz-juelich.de/api/v2'
    req = []
    for key, val in flags.items():
        req.append(f'{key}={val}')
    req = f'{pre}?' + '&'.join(req)
    req = f'{url}/{path}/{req}'

    if verbose:
        print(req)
    r = requests.get(req)
    if not r:
        if verbose:
            print(r.reason)
        return r
    else:
        return r.json()

ts = query('search', {
    'limit': None,
    'variable_id': 5,
    'fields': ','.join([
        'id',
        'data_start_date',
        'data_end_date',
    ])
})

start, end = '2017-01-01', '2017-12-31'
start, end = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')

ids = [record['id'] for record in ts if all([
    pd.Timestamp(record['data_start_date']) < start,
    end < pd.Timestamp(record['data_end_date'])
])]

print(f'Found {len(ids)} of {len(ts)} total records - {len(ids)/len(ts)*100:.2f}%')

dfs = []
for id in tqdm(ids, desc='Downloading TOAR2 data'):
    data = query(f'data/timeseries', {'limit': None}, pre=id, verbose=False)
    df   = pd.DataFrame(data['data'])
    df   = df.drop(columns=['timeseries_id', 'version'])
    df['lat'] = data['metadata']['station']['coordinates']['lat']
    df['lon'] = data['metadata']['station']['coordinates']['lng']
    df['datetime'] = pd.to_datetime(df.datetime)
    df.to_hdf('o3.2017.h5', str(id))

print('Done')
