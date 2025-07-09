import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import xarray as xr
from pymavlink import mavutil
import os
import sys
import time
from scipy.ndimage import uniform_filter1d
from datetime import datetime

# Constants for GPS time conversion
GPS_LEAP_SECONDS = 18  # Current difference between GPS and Unix time
GPS_TO_UNIX_OFFSET = 315964800  # Seconds between GPS epoch (1980-01-06) and Unix epoch (1970-01-01)


def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select ArduPilot DataFlash Log (.bin)",
        filetypes=[("DataFlash Logs", "*.bin"), ("All files", "*.*")]
    )
    return file_path


def process_dataflash_log(bin_file):
    print(f"Processing {os.path.basename(bin_file)}...")
    mlog = mavutil.mavlink_connection(bin_file)

    # Initialize data containers
    GPS_data = []
    IMU_data = []
    BARO_data = []
    TEMP_data = []
    HUM_data = []
    first_timestamp = None
    timezone_offset = 0  # Will be set based on first GPS message

    # First pass - set timezone offset from first GPS message
    mlog.rewind()
    while True:
        msg = mlog.recv_match()
        if msg is None:
            break

        if msg.get_type() == 'GPS' and hasattr(msg, 'GWk') and hasattr(msg, 'GMS'):
            # Calculate timezone offset from first GPS message
            gps_week_seconds = msg.GWk * 604800
            gps_ms_seconds = msg.GMS / 1000.0
            gps_epoch = gps_week_seconds + gps_ms_seconds
            unix_timestamp = gps_epoch + GPS_TO_UNIX_OFFSET - GPS_LEAP_SECONDS

            # Get local time from system to determine timezone offset
            local_time = datetime.fromtimestamp(unix_timestamp)
            utc_time = datetime.utcfromtimestamp(unix_timestamp)
            timezone_offset = (local_time - utc_time).total_seconds()
            break

    # Second pass - collect all data with absolute timestamps
    mlog.rewind()
    while True:
        try:
            msg = mlog.recv_match()
            if msg is None:
                break

            if msg.get_type() == 'GPS' and hasattr(msg, 'GWk') and hasattr(msg, 'GMS'):
                # Calculate absolute timestamp for GPS messages
                gps_week_seconds = msg.GWk * 604800
                gps_ms_seconds = msg.GMS / 1000.0
                gps_epoch = gps_week_seconds + gps_ms_seconds
                unix_timestamp = gps_epoch + GPS_TO_UNIX_OFFSET - GPS_LEAP_SECONDS - timezone_offset

                if hasattr(msg, 'Lat') and hasattr(msg, 'Lng') and hasattr(msg, 'Alt'):
                    GPS_data.append([
                        unix_timestamp,
                        msg.Lat,  # degrees
                        msg.Lng,
                        msg.Alt  # meters
                    ])

                    if first_timestamp is None or unix_timestamp < first_timestamp:
                        first_timestamp = unix_timestamp

            elif hasattr(msg, 'TimeUS'):
                # For non-GPS messages, use TimeUS relative to first timestamp
                if first_timestamp is not None:
                    abs_time = first_timestamp + (msg.TimeUS / 1e6)  # Convert to seconds and add to first timestamp

                    if msg.get_type() == 'IMU' and hasattr(msg, 'Temp'):
                        IMU_data.append([abs_time, msg.Temp])

                    elif msg.get_type() == 'BARO' and hasattr(msg, 'Press'):
                        temp = getattr(msg, 'Temp', np.nan)
                        BARO_data.append([abs_time, msg.Press, temp])

                    elif msg.get_type() == 'TEMP' and hasattr(msg, 'Temp1'):
                        TEMP_data.append([
                            abs_time,
                            msg.Temp1,
                            getattr(msg, 'Temp2', np.nan),
                            getattr(msg, 'Temp3', np.nan)
                        ])

                    elif msg.get_type() == 'HUM' and hasattr(msg, 'Humidity'):
                        HUM_data.append([
                            abs_time,
                            msg.Humidity,
                            getattr(msg, 'Temp', np.nan)
                        ])

        except Exception as e:
            print(f"Warning: Skipping message - {str(e)}")
            continue

    # Convert to numpy arrays
    GPS_data = np.array(GPS_data) if len(GPS_data) > 0 else np.empty((0, 4))
    IMU_data = np.array(IMU_data) if len(IMU_data) > 0 else np.empty((0, 2))
    BARO_data = np.array(BARO_data) if len(BARO_data) > 0 else np.empty((0, 3))
    TEMP_data = np.array(TEMP_data) if len(TEMP_data) > 0 else np.empty((0, 4))
    HUM_data = np.array(HUM_data) if len(HUM_data) > 0 else np.empty((0, 3))

    if len(GPS_data) == 0:
        print("Error: No valid GPS data found")
        return None

    # Create time reference from GPS data (absolute Unix timestamps)
    time_points = np.unique(np.floor(GPS_data[:, 0]))
    output_len = len(time_points)

    # Initialize output with observation numbers
    output_data = {
        'obs': np.arange(1, output_len + 1),  # 1-based index
        'lat': np.full(output_len, np.nan),
        'lon': np.full(output_len, np.nan),
        'altitude': np.full(output_len, np.nan),
        'time': time_points,  # Absolute Unix timestamps
        'air_temp': np.full(output_len, np.nan),
        'dew_point': np.full(output_len, np.nan),
        'rel_hum': np.full(output_len, np.nan),
        'air_press': np.full(output_len, np.nan),
        'gpt': np.full(output_len, np.nan),
        'gpt_height': np.full(output_len, np.nan),
        'wind_speed': np.full(output_len, np.nan),
        'wind_dir': np.full(output_len, np.nan)
    }

    # Use only raw latitude and longitude from GPS_0 (columns 9 and 10)
    for i, t in enumerate(time_points):
        # GPS_data[:, 0] is time; columns 1 and 2 should match lat/lon from columns 9 and 10 in original GPS_0
        mask = np.floor(GPS_data[:, 0]) == t
        if np.any(mask):
            # Take the first GPS_0 message at this second
            gps_row = GPS_data[mask][0]
            lat = gps_row[1]  # column 9 from original GPS_0
            lon = gps_row[2]  # column 10 from original GPS_0
            alt = gps_row[3]

            if -90 <= lat <= 90 and -180 <= lon <= 180:
                output_data['lat'][i] = lat
                output_data['lon'][i] = lon
                output_data['altitude'][i] = alt

    # Process barometric pressure
    if len(BARO_data) > 0:
        for i, t in enumerate(time_points):
            mask = np.floor(BARO_data[:, 0]) == t
            if np.any(mask):
                output_data['air_press'][i] = np.nanmean(BARO_data[mask, 1])

    # Process humidity data
    if len(HUM_data) > 0:
        for i, t in enumerate(time_points):
            mask = np.floor(HUM_data[:, 0]) == t
            if np.any(mask):
                output_data['rel_hum'][i] = np.nanmean(HUM_data[mask, 1])

    # Process temperature data (priority: TEMP > IMU > BARO > HUM)
    temp_sources = []
    if len(TEMP_data) > 0:
        temp_sources.append(TEMP_data[:, [0, 1]])  # Temp1
    if len(IMU_data) > 0:
        temp_sources.append(IMU_data)
    if len(BARO_data) > 0:
        temp_sources.append(BARO_data[:, [0, 2]])
    if len(HUM_data) > 0:
        temp_sources.append(HUM_data[:, [0, 2]])

    if temp_sources:
        all_temp_data = np.concatenate(temp_sources)
        for i, t in enumerate(time_points):
            mask = np.floor(all_temp_data[:, 0]) == t
            if np.any(mask):
                output_data['air_temp'][i] = np.nanmean(all_temp_data[mask, 1])

        # Apply moving average
        if output_len > 9:
            output_data['air_temp'] = uniform_filter1d(output_data['air_temp'], size=9, mode='nearest')

    return pd.DataFrame(output_data)


def save_outputs(data, base_name):
    if data is None or len(data) == 0:
        print("No valid data to save")
        return False

    # Save as text file
    txt_output = f"{base_name}.txt"
    with open(txt_output, 'w') as f:
        # Write header
        f.write("obs,lat,lon,altitude,time,air_temp,dew_point,rel_hum,air_press,gpt,gpt_height,wind_speed,wind_dir\n")

        # Write data with proper formatting
        for _, row in data.iterrows():
            line = [
                str(int(row['obs'])),
                f"{row['lat']:.7f}" if not np.isnan(row['lat']) else 'NaN',
                f"{row['lon']:.7f}" if not np.isnan(row['lon']) else 'NaN',
                f"{row['altitude']:.2f}" if not np.isnan(row['altitude']) else 'NaN',
                f"{row['time']:.2f}" if not np.isnan(row['time']) else 'NaN',  # Unix timestamp
                f"{row['air_temp']:.6f}" if not np.isnan(row['air_temp']) else 'NaN',
                f"{row['dew_point']:.6f}" if not np.isnan(row['dew_point']) else 'NaN',
                f"{row['rel_hum']:.6f}" if not np.isnan(row['rel_hum']) else 'NaN',
                f"{row['air_press']:.6f}" if not np.isnan(row['air_press']) else 'NaN',
                'NaN', 'NaN', 'NaN', 'NaN'  # Placeholders
            ]
            f.write(','.join(line) + '\n')

    print(f"Created text file: {txt_output}")

    # Save as NetCDF file
    nc_output = f"{base_name}.nc"
    try:
        # Convert Unix timestamps to datetime64 for NetCDF
        time_values = pd.to_datetime(data['time'], unit='s').values

        # Create dataset without attributes first
        ds = xr.Dataset(
            {
                'latitude': ('time', data['lat'].values),
                'longitude': ('time', data['lon'].values),
                'altitude': ('time', data['altitude'].values),
                'air_temperature': ('time', data['air_temp'].values),
                'dew_point_temperature': ('time', data['dew_point'].values),
                'relative_humidity': ('time', data['rel_hum'].values),
                'air_pressure': ('time', data['air_press'].values)
            },
            coords={
                'time': ('time', time_values),
                'observation': ('time', data['obs'].values)
            }
        )

        # Now add attributes carefully
        variable_attrs = {
            'time': {
                'long_name': 'Time',
                'standard_name': 'time',
                'comment': 'Absolute time in UTC'
            },
            'latitude': {
                'units': 'degrees_north',
                'long_name': 'Latitude',
                'standard_name': 'latitude'
            },
            'longitude': {
                'units': 'degrees_east',
                'long_name': 'Longitude',
                'standard_name': 'longitude'
            },
            'altitude': {
                'units': 'meters',
                'long_name': 'Altitude above mean sea level',
                'standard_name': 'altitude',
                'positive': 'up'
            },
            'air_temperature': {
                'units': 'degree_Celsius',
                'long_name': 'Air temperature',
                'standard_name': 'air_temperature'
            },
            'dew_point_temperature': {
                'units': 'degree_Celsius',
                'long_name': 'Dew point temperature',
                'standard_name': 'dew_point_temperature'
            },
            'relative_humidity': {
                'units': 'percent',
                'long_name': 'Relative humidity',
                'standard_name': 'relative_humidity'
            },
            'air_pressure': {
                'units': 'hPa',
                'long_name': 'Air pressure',
                'standard_name': 'air_pressure'
            }
        }

        # Apply attributes
        for var_name, attrs in variable_attrs.items():
            if var_name in ds.variables:
                ds[var_name].attrs.update(attrs)

        # Set encoding for time variable to prevent conflicts
        encoding = {
            'time': {
                'dtype': 'double',
                '_FillValue': None,
                'units': 'seconds since 1970-01-01 00:00:00'
            }
        }

        ds.to_netcdf(nc_output, encoding=encoding)
        print(f"Created NetCDF file: {nc_output}")
        return True

    except Exception as e:
        print(f"Error creating NetCDF file: {str(e)}")
        return False


if __name__ == "__main__":
    print("ArduPilot DataFlash Log Processor")
    input_file = select_file()

    if input_file:
        try:
            data = process_dataflash_log(input_file)
            if data is not None:
                base_name = os.path.splitext(input_file)[0]
                if save_outputs(data, base_name):
                    print("Processing completed successfully")
                else:
                    print("Processing completed with output errors")
            else:
                print("No valid data extracted from log file")
        except Exception as e:
            print(f"Fatal processing error: {str(e)}")

    if sys.platform.startswith('win'):
        time.sleep(10)
    else:
        input("Press Enter to exit...")