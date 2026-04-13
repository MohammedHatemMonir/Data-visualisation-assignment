import pandas as pd
import numpy as np
import os

def process_weather_data(input_file):
    # 1. Load Data
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found in current directory.")
        return
    
    df = pd.read_csv(input_file)
    print(f"Initial row count: {len(df)}")

    # 2. Pre-processing: Remove rows with any empty fields
    # This cleans the 10,000+ rows with missing location/device names
    df = df.dropna().reset_index(drop=True)
    print(f"Cleaned row count: {len(df)}")

    # 3. Temporal Feature Extraction
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.day_name()
    # Define night as 8 PM to 6 AM
    df['is_night'] = df['hour'].apply(lambda x: 1 if x >= 20 or x <= 6 else 0)

    # 4. Spatial Feature Extraction (Split Latitude/Longitude)
    # Splits "-38.2391476, 144.3387083" into two floats
    df[['latitude', 'longitude']] = df['device_location'].str.split(',', expand=True).astype(float)

    # 5. Meteorological Derived Features
    # Dew Point Calculation (Magnus-Tetens Formula)
    b, c = 17.67, 243.5
    def calculate_dew_point(T, RH):
        gamma = np.log(RH/100) + (b * T) / (c + T)
        return (c * gamma) / (b - gamma)

    df['dew_point'] = calculate_dew_point(df['temperature_merged'], df['humidity_merged'])

    # Heat Index / Humidex (Simplified formula for localized comfort)
    # Humidex = T + 0.5555 * (e - 10) where e is vapor pressure
    e = 6.11 * np.exp(5417.7530 * ((1/273.16) - (1/(df['dew_point'] + 273.16))))
    df['humidex'] = df['temperature_merged'] + 0.5555 * (e - 10)

    # 6. Time-Series Engineering (Rolling Averages)
    # Group by device to ensure we aren't averaging different locations together
    df = df.sort_values(['device_id', 'time'])
    df['temp_rolling_1h'] = df.groupby('device_id')['temperature_merged'].transform(lambda x: x.rolling(4, min_periods=1).mean())

    # 7. Save to New File
    output_file = "processed_weather_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Successfully saved {len(df)} rows to {output_file}")

if __name__ == "__main__":
    process_weather_data('weather-together-temperature-and-humidity.csv')