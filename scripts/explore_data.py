import pandas as pd
import numpy as np

df = pd.read_csv('jan to may police violation_anonymized791b166.csv', low_memory=False)
print('=== SHAPE ===')
print(df.shape)
print('\n=== COLUMNS ===')
print(list(df.columns))
print('\n=== DTYPES ===')
print(df.dtypes)
print('\n=== NULL COUNTS ===')
print(df.isnull().sum())
print('\n=== VEHICLE TYPE VALUE COUNTS ===')
print(df['vehicle_type'].value_counts())
print('\n=== VIOLATION TYPE VALUE COUNTS ===')
print(df['violation_type'].value_counts().head(15))
print('\n=== POLICE STATION VALUE COUNTS ===')
print(df['police_station'].value_counts().head(20))
print('\n=== VALIDATION STATUS ===')
print(df['validation_status'].value_counts())
print('\n=== DATA SENT TO SCITA ===')
print(df['data_sent_to_scita'].value_counts())
print('\n=== DATE RANGE ===')
df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce')
min_d = df['created_datetime'].min()
max_d = df['created_datetime'].max()
print('Min date:', min_d)
print('Max date:', max_d)
print('\n=== LAT/LONG RANGE ===')
lat_min = df['latitude'].min()
lat_max = df['latitude'].max()
lon_min = df['longitude'].min()
lon_max = df['longitude'].max()
print('Lat:', lat_min, 'to', lat_max)
print('Lon:', lon_min, 'to', lon_max)
print('\n=== JUNCTION NAME VALUE COUNTS ===')
print(df['junction_name'].value_counts().head(15))
print('\n=== UNIQUE COUNTS ===')
for col in ['location','police_station','vehicle_type','junction_name','offence_code','validation_status']:
    print(col, ':', df[col].nunique(), 'unique values')

print('\n=== OFFENCE CODE VALUE COUNTS ===')
print(df['offence_code'].value_counts().head(15))

print('\n=== CLOSED_DATETIME NON-NULL ===')
print(df['closed_datetime'].notna().sum(), 'out of', len(df))

print('\n=== ACTION_TAKEN_TIMESTAMP NON-NULL ===')
print(df['action_taken_timestamp'].notna().sum(), 'out of', len(df))

# Check time deltas where possible
df['closed_datetime'] = pd.to_datetime(df['closed_datetime'], errors='coerce')
valid_both = df.dropna(subset=['created_datetime','closed_datetime'])
if len(valid_both) > 0:
    valid_both = valid_both.copy()
    valid_both['duration_hours'] = (valid_both['closed_datetime'] - valid_both['created_datetime']).dt.total_seconds() / 3600
    print('\n=== VIOLATION DURATION (hours) ===')
    print(valid_both['duration_hours'].describe())
else:
    print('\nNo rows with both created and closed datetime')

print('\n=== UPDATED VEHICLE TYPE ===')
print(df['updated_vehicle_type'].value_counts().head(15))

# Hourly distribution
df['hour'] = df['created_datetime'].dt.hour
print('\n=== HOURLY DISTRIBUTION ===')
print(df['hour'].value_counts().sort_index())
