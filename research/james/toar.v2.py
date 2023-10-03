
#%%
#%%

import requests



requests.get(f'{url}/variables/o3').json()


#%%
path = 'data/timeseries'
flags = {
    # 'limit': '2',
    'data_start_date': '2017-01-01',
    # 'data_end_date': '2017-12-31'
}

req = []
for key, val in flags.items():
    req.append(f'{key}={val}')
req = '?' + '&'.join(req)
req = f'{url}/{path}/{req}'
if req:
    req

r = requests.get(req)
if not r:
    print(r.reason)
data = r.json()
data

#%%



#%%

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

#%%

query('timeseries', {
    'limit': None,
    # 'data_start_date': '2017-01-01',
    # 'data_end_date': '2017-12-31'
    'variable_id': 5
})

#%%
key = 'o3'
for var in variables:
    if var['name'] == key:
        break

ts = query('search', {
    'limit': None,
    # 'data_start_date': '2017-01-01',
    'variable_id': var['id'],
    'fields': ','.join([
        'id',
        'data_start_date',
        'data_end_date',
    ])
})
ts

#%%
import pandas as pd

start, end = '2017-01-01', '2017-12-31'
start, end = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')

ids = [record['id'] for record in ts if all([
    pd.Timestamp(record['data_start_date']) < start,
    end < pd.Timestamp(record['data_end_date'])
])]

print(f'Found {len(ids)} of {len(ts)} total records - {len(ids)/len(ts)*100:.2f}%')

#%%
from tqdm import tqdm

dfs = []
for id in tqdm(ids, desc='Downloading TOAR2 data'):
    data = query(f'data/timeseries', {'limit': None}, pre=id, verbose=False)
    df   = pd.DataFrame(data['data'])
    df   = df.drop(columns=['timeseries_id', 'version'])
    df['lat'] = data['metadata']['station']['coordinates']['lat']
    df['lon'] = data['metadata']['station']['coordinates']['lng']
    df['datetime'] = pd.to_datetime(df.datetime)
    df.to_hdf('o3.2017.h5', str(id))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
import requests

#%%
base = 'https://toar-data.fz-juelich.de/api/v2'
url = f'{base}/stationmeta/?limit=None&fields=name,coordinates'
url
req = requests.get(url)
data = req.json()
data
len(data)

#%%


.replace(':', '%3A').replace('+', '%2B')
'https://toar-data.fz-juelich.de/api/v2/timeseries/?limit=1'

#%%

query = {
    'limit'    : 'None',
    'data_start_date': '2017-01-01T00:00:00+0000',
    'data_end_date'  : '2018-01-01T00:00:00+0000',
    'parameter': 'o3',
    'order_by' : 'datetime'
}
params = ('&'.join([f'{k}={v}' for k, v in query.items()])).replace(':', '%3A').replace('+', '%2B')

https://toar-data.fz-juelich.de/api/v2/data/timeseries/52

#%%
import requests

from mlky import Section


def query(url, api, flags={}, pre='', timeout=5, verbose=True):
    """
    """
    params = ''
    if flags:
        params = '?' + ('&'.join([f'{k}={v}' for k, v in flags.items()])).replace(':', '%3A').replace('+', '%2B')
    request = f'{url}/{api}/{pre}{params}'

    if verbose:
        print(f'request({request!r})')

    try:
        r = requests.get(request, timeout=timeout)
    except:
        return
    if not r:
        if verbose:
            print(f'Request failed, reason: {r.reason}')
    return r


# Base URL for TOAR API v2
url = 'https://toar-data.fz-juelich.de/api/v2'

# Retrieve the o3 ID
req = query(url, 'variables', pre='o3')
o3  = req.json()['id']

# Retrieve the list of stations
stations = Section(
    name = 'Stations',
    data = query(url, 'stationmeta', {
        'limit' : None,
        'fields': ','.join([
            'id',
            'name',
            'coordinates'
        ])
    }).json()
)

#%%
# Reduce the stations list to those that are helpful
for _, station in stations.items():
    break

#%%

r = query(url, 'timeseries', {
    'limit'      : None,
    'station_id' : station.id,
    'variable_id': o3,
    'fields'     : ','.join([
        'id',
        'data_start_date',
        'data_end_date'
    ])
}).json()
len(r)

ts = r[0]
ts.keys()
ts['id'], ts['data_start_date'], ts['data_end_date']

#%%

r = query(url, 'data/timeseries', pre=3, flags={
    'format': 'csv'
})
# print(r.text[0])
r.text
# r.json()

#%%
import pandas as pd

ts = query(url, 'search', {
    'limit': None,
    'variable_id': o3,
    'fields': ','.join([
        'id',
        'data_start_date',
        'data_end_date',
    ])
}).json()

start, end = '2017-01-01', '2017-12-31'
start, end = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')

# ids = [record['id'] for record in ts if all([
#     pd.Timestamp(record['data_start_date']) < start,
#     end < pd.Timestamp(record['data_end_date'])
# ])]
ids = []
for record in ts:
    dstart = pd.Timestamp(record['data_start_date'])
    dend   = pd.Timestamp(record['data_end_date'])
    if any([
        ( start <= dstart) & (dstart <=  end), # Starts in range
        ( start <= dend)   & (  dend <=  end), # Ends in range
        (dstart <=  start) & (   end <= dend)  # Spans range
    ]):
        ids.append(record['id'])

ids.sort()
print(f'Found {len(ids)} of {len(ts)} total records - {len(ids)/len(ts)*100:.2f}%')

#%%
import json

from tqdm import tqdm

failed = []
for id in tqdm(ids, desc='Downloading TOAR2 data'):
    req = query(url, f'data/timeseries', pre=id, verbose=False, timeout=5, flags={
        'format': 'csv'
        }
    )
    if req:
        data = req.text

        df = pd.read_csv(StringIO(data),
            comment     = '#',
            header      = 1,
            sep         = ',',
            names       = ["time", "value", "flags", "version", "timeseries_id"],
            parse_dates = ["time"],
            index_col   = "time",
            date_parser = lambda x:pd.to_datetime(x,utc=True),infer_datetime_format=True
        ).drop(columns=['flags', 'version'])

        # Extract the metadata
        meta = json.loads("\n".join([line[1:] for line in data.split('\n') if line.startswith('#')]))
        df['lat'] = meta['station']['coordinates']['lat']
        df['lon'] = meta['station']['coordinates']['lng']

        df.to_hdf('o3.2017.h5', str(id))
        del df
    else:
        failed.append(id)

#%%

help(requests.get)

#%%





#%%

dfs = []
for id in tqdm(ids, desc='Downloading TOAR2 data'):
    data = query(f'data/timeseries', {'limit': None}, pre=id, verbose=False)
    df   = pd.DataFrame(data['data'])
    df   = df.drop(columns=['timeseries_id', 'version'])
    df['lat'] = data['metadata']['station']['coordinates']['lat']
    df['lon'] = data['metadata']['station']['coordinates']['lng']
    df['datetime'] = pd.to_datetime(df.datetime)
    df.to_hdf('o3.2017.h5', str(id))



#%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
import requests

from tqdm import tqdm

def query(page=1, limit=1000):
    """
    """
    base  = "https://api.openaq.org/v2"
    sect  = 'measurements'
    query = {
        'limit'    : limit,
        'page'     : page,
        'offset'   : 0,
        'date_from': '2017-01-01T00:00:00+0000',
        'date_to'  : '2018-01-01T00:00:00+0000',
        'has_geo'  : 'true',
        'parameter': 'o3',
        'order_by' : 'datetime'
    }
    params = ('&'.join([f'{k}={v}' for k, v in query.items()])).replace(':', '%3A').replace('+', '%2B')

    return requests.get(f'{base}/{sect}?{params}')

R = None
def iterate(total):
    """
    """
    i = 0
    while True:
        i += 1
        req = query(page=i)
        if req:
            for data in req.json()['results']:
                yield data

        else:
            if i < total:
                print(f"Bad page {i}, reason: {req.text}")
                global R
                R = req
            else:

                break

data  = query(limit=1).json()
total = data['meta']['found']

rows = []
keys = ['value', 'lat', 'lon', 'time', 'local', 'unit', 'locationId', 'country', 'city']

#%%

for data in tqdm(iterate(total), total=total):
    data['time']  = data['date']['utc']
    data['local'] = data['date']['local']
    coords = data['coordinates']
    if coords:
        data['lat'] = coords['latitude']
        data['lon'] = coords['longitude']

        rows.append([data[key] for key in keys])

R.text

#%%
