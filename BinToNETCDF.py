#!/usr/bin/env python3
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

GPS_LEAP_SECONDS = 18
GPS_TO_UNIX_OFFSET = 315964800

def compute_dew_point(Tc, RH):
    """
    Magnus formula: Tc in °C, RH in percent → dew point in °C
    """
    a, b = 17.62, 243.12
    RH = np.clip(RH, 0.1, 100.0)
    α = (a * Tc) / (b + Tc) + np.log(RH / 100.0)
    return (b * α) / (a - α)

def select_file():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select ArduPilot DataFlash Log (.bin)",
        filetypes=[("DataFlash Logs", "*.bin"), ("All files", "*.*")]
    )

def process_dataflash_log(bin_file):
    print(f"Processing {os.path.basename(bin_file)} …")
    mlog = mavutil.mavlink_connection(
        bin_file,
        dialect='ardupilotmega',
        robust_parsing=True
    )

    GPS_data, BARO_data, TEMP_data, HUM_data, IMU_data = [], [], [], [], []

    while True:
        msg = mlog.recv_match()
        if msg is None:
            break
        ts = getattr(msg, '_timestamp', None)
        if ts is None:
            continue
        typ = msg.get_type()
        if typ == 'GPS' and hasattr(msg, 'Lat') and hasattr(msg, 'Lng') and hasattr(msg, 'Alt'):
            GPS_data.append([ts, msg.Lat, msg.Lng, msg.Alt])
        elif typ == 'WXTP':
            d = msg.to_dict()
            temps_k = [d.get(' t0', np.nan), d.get(' t1', np.nan), d.get(' t2', np.nan)]
            temps_c = [t - 273.15 for t in temps_k]
            TEMP_data.append([ts, *temps_c])
        elif typ == 'WXRH':
            d = msg.to_dict()
            rh_vals = [d.get(' rh0', np.nan), d.get(' rh1', np.nan), d.get(' rh2', np.nan)]
            tmp_k = [d.get(' t0', np.nan), d.get(' t1', np.nan), d.get(' t2', np.nan)]
            tmp_c = [t - 273.15 for t in tmp_k]
            HUM_data.append([ts, np.nanmean(rh_vals), np.nanmean(tmp_c)])
        elif typ in ('SCALED_PRESSURE', 'BARO'):
            p = getattr(msg, 'press_abs', None) or getattr(msg, 'Press', None)
            t = getattr(msg, 'temperature', None) or getattr(msg, 'Temp', np.nan)
            if p is not None:
                t_c = (t - 273.15) if t > 200 else t
                BARO_data.append([ts, p, t_c])
        elif typ in ('TEMP', 'TEMPERATURE'):
            a = getattr(msg, 'Temp1', np.nan)
            b = getattr(msg, 'Temp2', np.nan)
            c = getattr(msg, 'Temp3', np.nan)
            TEMP_data.append([ts, a, b, c])
        elif typ == 'IMU' and hasattr(msg, 'Temp'):
            IMU_data.append([ts, msg.Temp])

    GPS_data = np.array(GPS_data) if GPS_data else np.empty((0, 4))
    BARO_data = np.array(BARO_data) if BARO_data else np.empty((0, 3))
    TEMP_data = np.array(TEMP_data) if TEMP_data else np.empty((0, 4))
    HUM_data = np.array(HUM_data) if HUM_data else np.empty((0, 3))
    IMU_data = np.array(IMU_data) if IMU_data else np.empty((0, 2))

    if GPS_data.size == 0:
        print("Error: no GPS records found.")
        return None

    bins = np.unique(np.floor(GPS_data[:, 0]))
    N = len(bins)
    out = {
        'obs': np.arange(1, N + 1),
        'lat': np.full(N, np.nan),
        'lon': np.full(N, np.nan),
        'altitude': np.full(N, np.nan),
        'time': bins,
        'air_temp': np.full(N, np.nan),
        'dew_point': np.full(N, np.nan),
        'rel_hum': np.full(N, np.nan),
        'air_press': np.full(N, np.nan),
        'gpt': np.full(N, np.nan),
        'gpt_height': np.full(N, np.nan),
        'wind_speed': np.full(N, np.nan),
        'wind_dir': np.full(N, np.nan),
    }

    for i, t in enumerate(bins):
        m = np.floor(GPS_data[:, 0]) == t
        if m.any():
            out['lat'][i], out['lon'][i], out['altitude'][i] = GPS_data[m][0, 1:4]

    if BARO_data.size:
        for i, t in enumerate(bins):
            m = np.floor(BARO_data[:, 0]) == t
            if m.any():
                out['air_press'][i] = np.nanmean(BARO_data[m, 1])

    if HUM_data.size:
        for i, t in enumerate(bins):
            m = np.floor(HUM_data[:, 0]) == t
            if m.any():
                out['rel_hum'][i] = np.nanmean(HUM_data[m, 1])

    temp_sources = []
    if TEMP_data.size:
        temp_sources.append(TEMP_data[:, [0, 1]])
    if IMU_data.size:
        temp_sources.append(IMU_data)
    if BARO_data.size:
        temp_sources.append(BARO_data[:, [0, 2]])
    if HUM_data.size:
        temp_sources.append(HUM_data[:, [0, 2]])

    if temp_sources:
        all_t = np.vstack(temp_sources)
        for i, t in enumerate(bins):
            m = np.floor(all_t[:, 0]) == t
            if m.any():
                out['air_temp'][i] = np.nanmean(all_t[m, 1])
        if N > 9:
            out['air_temp'] = uniform_filter1d(out['air_temp'], size=9, mode='nearest')

    out['dew_point'] = compute_dew_point(out['air_temp'], out['rel_hum'])
    return pd.DataFrame(out)

def save_outputs(df, base):
    if df is None or df.empty:
        print("No data to save.")
        return False

    txt_file = f"{base}.txt"
    df.to_csv(txt_file, index=False, columns=[
        'obs', 'lat', 'lon', 'altitude', 'time',
        'air_temp', 'dew_point', 'rel_hum',
        'air_press', 'gpt', 'gpt_height', 'wind_speed', 'wind_dir'
    ])
    print(f"Written: {txt_file}")

    nc_file = f"{base}.nc"
    try:
        times = pd.to_datetime(df['time'], unit='s')
        ds = xr.Dataset(
            {
                'latitude': ('time', df['lat']),
                'longitude': ('time', df['lon']),
                'altitude': ('time', df['altitude']),
                'air_temperature': ('time', df['air_temp']),
                'dew_point_temperature': ('time', df['dew_point']),
                'relative_humidity': ('time', df['rel_hum']),
                'air_pressure': ('time', df['air_press']),
            },
            coords={
                'time': ('time', times),
                'observation': ('time', df['obs'])
            }
        )
        attrs = {
            'latitude': {'units': 'degrees_north', 'standard_name': 'latitude'},
            'longitude': {'units': 'degrees_east', 'standard_name': 'longitude'},
            'altitude': {'units': 'm', 'standard_name': 'altitude', 'positive': 'up'},
            'air_temperature': {'units': '°C', 'standard_name': 'air_temperature'},
            'dew_point_temperature': {'units': '°C', 'standard_name': 'dew_point_temperature'},
            'relative_humidity': {'units': '%', 'standard_name': 'relative_humidity'},
            'air_pressure': {'units': 'hPa', 'standard_name': 'air_pressure'},
        }
        for v, a in attrs.items():
            ds[v].attrs.update(a)
        ds.to_netcdf(nc_file, encoding={'time': {'dtype': 'double', 'units': 'seconds since 1970-01-01'}})
        print(f"Written: {nc_file}")
        return True
    except Exception as e:
        print("Error writing NetCDF:", e)
        return False

if __name__ == "__main__":
    print("ArduPilot DataFlash → TXT/NetCDF")
    infile = select_file()
    if infile:
        df = process_dataflash_log(infile)
        if df is not None:
            base = os.path.splitext(infile)[0]
            ok = save_outputs(df, base)
            print("Done" if ok else "Done with errors")
    if sys.platform.startswith('win'):
        time.sleep(5)
    else:
        input("Press Enter to exit…")