import pandas as pd
import os

## IMPORTING THE AEMO DATA

# relative directory path
directory = './data/demand'
dfs = [] 

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        # Create the full file path and read the file
        file_path = os.path.join(directory, filename)
        df_temp = pd.read_csv(file_path)
        
        # FIX THE FIRST ROW IN THE LOOP
        df_temp['SETTLEMENTDATE'] = pd.to_datetime(df_temp['SETTLEMENTDATE'])
        if not df_temp.empty:
            df_temp.loc[0, 'SETTLEMENTDATE'] -= pd.Timedelta(minutes=5)
        
        # Check if column exists to prevent errors
        print(f"File: {filename}")
        dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True)
df.drop(columns=['REGION', 'PERIODTYPE'], inplace=True)
df.rename(columns={'SETTLEMENTDATE': 'DATE_TIME', 'TOTALDEMAND': 'TOTAL_DEMAND'}, inplace=True, errors='ignore')

df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
df = df[df['DATE_TIME'].dt.minute.isin([0, 30])].copy()
df = df.sort_values('DATE_TIME').reset_index(drop=True)

## CREATING COLUMNS FOR SCHOOL_HOLIDAY, PUBLIC_HOLIDAY
import holidays

# Convert to Pandas datetime for better parsing
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])

nsw_holidays = holidays.Australia(subdiv='NSW', years=range(2021, 2027))
df['PUBLIC_HOLIDAY'] = df['DATE_TIME'].dt.date.isin(nsw_holidays)

school_holiday_ranges = [
    # 2022
    ('2022-01-01', '2022-01-27'), # Summer 21-22 end
    ('2022-04-11', '2022-04-22'), # Autumn
    ('2022-07-04', '2022-07-15'), # Winter
    ('2022-09-26', '2022-10-07'), # Spring
    ('2022-12-21', '2022-12-31'), # Summer 22-23 start

    # 2023
    ('2023-01-01', '2023-01-26'), # Summer 22-23
    ('2023-04-10', '2023-04-21'), # Autumn
    ('2023-07-03', '2023-07-14'), # Winter
    ('2023-09-25', '2023-10-06'), # Spring
    ('2023-12-20', '2023-12-31'), # Summer 23-24 start

    # 2024
    ('2024-01-01', '2024-01-29'), # Summer 23-24 end
    ('2024-04-15', '2024-04-26'), # Autumn
    ('2024-07-08', '2024-07-19'), # Winter
    ('2024-09-30', '2024-10-11'), # Spring
    ('2023-12-23', '2023-12-31'), # Summer 24-25 start

    # 2025
    ('2025-01-01', '2025-01-30'), # Summer 24-25 end
    ('2025-04-14', '2025-04-24'), # Autumn
    ('2025-07-07', '2025-07-18'), # Winter
    ('2025-09-29', '2025-10-10'), # Spring
    ('2025-12-22', '2025-12-31'), # Summer 25-26 start

    # 2026
    ('2026-01-01', '2026-01-26'), # Summer 25-26 end
    ('2026-04-07', '2026-04-17'), # Autumn
    ('2026-07-06', '2026-07-17'), # Winter
    ('2026-09-28', '2026-10-09'), # Spring
    ('2026-12-18', '2026-12-31'), # Summer 26-27 start
]

holiday_dates = set()
for start, end in school_holiday_ranges:
    holiday_dates.update(pd.date_range(start, end).date)

df['SCHOOL_HOLIDAY'] = df['DATE_TIME'].dt.date.isin(holiday_dates) # check and store boolean


## WEATHER DATA
import pandas as pd
import datetime

def fetch_sydney_weather(start_year, end_year):
    # Sydney Airport Station ID is usually 'YSSY' in the ASOS network
    station = 'YSSY' 
    start_date = f"{start_year}-01-01+00:00"
    end_date = f"{end_year}-12-31+23:59"
    
    # Web scraping the global archive - added 'feel' and 'p01m' for transpiration context
    url = (f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
           f"station={station}&data=tmpc&data=dwpc&data=relh&data=feel&data=sknt&"
           f"data=skyc1&data=skyc2&data=skyc3&"
           f"year1={start_year}&month1=1&day1=1&"
           f"year2={end_year}&month2=12&day2=31&"
           f"tz=Etc%2FGMT-10&format=onlycomma&latlon=no&missing=M")
    
    print("Fetching weather data")
    weather_df = pd.read_csv(url, skiprows=0, na_values='M')
    
    # Rename columns for clarity
    weather_df = weather_df.rename(columns={
        'valid': 'DATE_TIME',
        'tmpc': 'TEMPERATURE',
        'relh': 'HUMIDITY',
        'dwpc': 'DWPC',
        'feel': 'APPARENT_TEMP',
        'sknt': 'WIND_SPEED',
        'skyc1': 'CLOUD_L1',
        'skyc2': 'CLOUD_L2',
        'skyc3': 'CLOUD_L3'
    })
    
    weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'])
    weather_df['APPARENT_TEMP'] = ((weather_df['APPARENT_TEMP'] - 32) * 5/9) # Farenheit to Celcius
    weather_df['WIND_SPEED'] = (weather_df['WIND_SPEED'] * 1.852) # Convert Wind Speed from knots to km/h (1 knot ≈ 1.852 km/h)
    
    return weather_df

weather_df = fetch_sydney_weather(2022, 2026)

# Cleaning dataset
cloud_map = {'CLR': 0, 'SKC': 0, 'FEW': 2, 'SCT': 4, 'BKN': 6, 'OVC': 8}
for col in ['CLOUD_L1', 'CLOUD_L2', 'CLOUD_L3']:
    weather_df[col] = weather_df[col].map(cloud_map).fillna(0).round()

weather_df['CLOUD_COVER'] = weather_df[['CLOUD_L1', 'CLOUD_L2', 'CLOUD_L3']].max(axis=1)

weather_df.drop(['station', 'CLOUD_L1', 'CLOUD_L2', 'CLOUD_L3'], axis=1, inplace=True, errors='ignore')

if 'DATE_TIME' in df.columns:
    demand_df = df.set_index('DATE_TIME').sort_index()
else:
    demand_df = df.sort_index()

if 'DATE_TIME' in weather_df.columns:
    weather_df = weather_df.set_index('DATE_TIME').sort_index()
else:
    weather_df = weather_df.sort_index()

weather_numeric = weather_df.select_dtypes(include=['number', 'float', 'int'])
weather_5min = weather_numeric.reindex(demand_df.index)

# Interpolate the gaps (Linear interpolation)
weather_5min = weather_5min.interpolate(method='linear')
weather_5min = weather_5min.bfill().ffill()

# merge back together
demand_df = demand_df.drop(columns=weather_5min.columns.intersection(demand_df.columns))
df = pd.concat([demand_df, weather_5min], axis=1).reset_index()

if 'index' in df.columns:
    df = df.rename(columns={'index': 'DATE_TIME'})


### Consumer spending and confidence


## RBA Monthly Cash Rate
cash_rate_df = pd.read_csv('./data/rba_monthly_cash_rate.csv')
df['Month_Year'] = df['DATE_TIME'].dt.strftime('%Y-%m') # parse date format like '2023-01'
df = df.merge(cash_rate_df, left_on='Month_Year', right_on='DATE', how='left') # broadcast join
df = df.drop(columns=['Month_Year', 'DATE']) # clean columns

## Monthly Household Spending Indicator (MHSI)
spending_df = pd.read_csv('./data/spending_millions.csv')
df['join_key'] = df['DATE_TIME'].dt.strftime('%b-%y')

# Merge and clean up
df = df.merge(spending_df, left_on='join_key', right_on='Month', how='left')
df = df.drop(columns=['join_key', 'Month'])

# Consumer Confidence (Westpac)
consumer_conf = pd.read_csv('./data/consumer_conf.csv')
df['month_key'] = df['DATE_TIME'].dt.strftime('%Y-%m')
df = df.merge(consumer_conf, left_on='month_key', right_on='year-month', how='left')
df = df.drop(columns=['month_key', 'year-month'])

pop_df = pd.read_csv('./data/population.csv')
pop_df['DATE_TIME'] = pd.to_datetime(pop_df['DATE_TIME'])

df = pd.merge(df, pop_df, on='DATE_TIME', how='left')
df['NSW_POPULATION'] = df['NSW_POPULATION'].interpolate(method='linear').round().astype(int)

from nemosis import dynamic_data_compiler
import pandas as pd

# Define project range
start_time = '2022/01/01 00:00:00'
end_time = '2026/02/01 00:00:00'

solar_df = dynamic_data_compiler(
    start_time=start_time,
    end_time=end_time,
    table_name='ROOFTOP_PV_ACTUAL',
    raw_data_location='./data/cache'  # This is the required positional argument
)

# Filter for NSW and clean up
solar_nsw = solar_df[solar_df['REGIONID'] == 'NSW1'].copy()

# AEMO ROOFTOP_PV_ACTUAL usually uses 'INTERVAL_DATETIME' or 'SETTLEMENTDATE'
date_col = 'INTERVAL_DATETIME' if 'INTERVAL_DATETIME' in solar_nsw.columns else 'SETTLEMENTDATE'

solar_nsw = solar_nsw[[date_col, 'POWER']].rename(
    columns={date_col: 'DATE_TIME', 'POWER': 'ROOFTOP_SOLAR_MW'}
)

solar_nsw['DATE_TIME'] = pd.to_datetime(solar_nsw['DATE_TIME'])

# Aggregate duplicates to prevent row-doubling
solar_nsw = solar_nsw.groupby('DATE_TIME')['ROOFTOP_SOLAR_MW'].mean().reset_index()

# Perform the merge
df = pd.merge(df, solar_nsw[['DATE_TIME', 'ROOFTOP_SOLAR_MW']], on='DATE_TIME', how='left')
df['ROOFTOP_SOLAR_MW'] = df['ROOFTOP_SOLAR_MW'].bfill(limit=1)

# Standard linear interpolation for any other gaps
df['ROOFTOP_SOLAR_MW'] = df['ROOFTOP_SOLAR_MW'].interpolate(method='linear')

# Zero out nighttime values
df.loc[(df['DATE_TIME'].dt.hour >= 20) | (df['DATE_TIME'].dt.hour <= 5), 'ROOFTOP_SOLAR_MW'] = 0
df['ROOFTOP_SOLAR_MW'] = df['ROOFTOP_SOLAR_MW'].fillna(0)

df['GROSS_DEMAND_MW'] = df['TOTAL_DEMAND'] + df['ROOFTOP_SOLAR_MW']

# round and save
df = df.round(4)
df.to_csv('./data/dataset.csv', index=False)