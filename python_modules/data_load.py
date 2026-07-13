from pathlib import Path
import xarray as xr
import pandas as pd

CLOUD_PATH = Path('/Users/markcampmier/Library/Mobile Documents/com~apple~CloudDocs/samosa_phase1/data/4_merged/')

ds = xr.open_dataset(CLOUD_PATH / 'collocation.nc')

df = ds[['pa_raw', 'rh', 'pm25']].where(ds.pm25_flag==0, drop=True).to_dataframe().drop(
    ['latitude', 'longitude', 'season', 'district', 'state', 'settlement_name'], axis=1)

df = df.reset_index().set_index('time').dropna()

print(f'DataFrame columns: {df.columns}')
print(f'Record count per site: {df.groupby('site').count().min(axis=1)}')