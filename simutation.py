# Visual Computing: Localization by Multilateration
# Simulation of beacon RSSI measurement by device
# Volker Ahlers, Hochschule Hannover, 2021

import pickle
import numpy as np
import pandas as pd
import minvc as vc

def create_measurement(device_name, device_location, beacons, n_signals=None, noise=0.0, \
                       keep_dists=True, keep_actual_location=True):
    # convert beacon names and locations to NumPy arrays
    beacon_names = beacons['name'].values
    beacon_locations = beacons[['x', 'y', 'z']].values
    # print(beacon_names, "\n\n", beacon_locations)

    # compute distances between device and beacons (2D for visualization only)
    n = beacon_names.size
    beacon_dists = np.linalg.norm(beacon_locations - device_location, axis=1) \
                   + np.random.normal(scale=noise, size=n)

    beacon_dists2d = np.linalg.norm(beacon_locations[:, 0:2] - device_location[0:2], axis=1) \
                     + np.random.normal(scale=noise, size=n)
    # print(beacon_dists)
    # print(beacon_dists2d)

    # compute RSSI values of 3D distances
    rssi_conv = vc.RSSIConverter()
    beacon_rssis = rssi_conv.get_rssi(beacon_dists)

    # create data frame, only keep n_signals strongest signals (if set)
    if keep_dists:
        beacon_data = pd.DataFrame({'name': beacon_names, 'rssi': beacon_rssis, \
                                    'dist': beacon_dists, 'dist2d': beacon_dists2d})
    else:
        beacon_data = pd.DataFrame({'name': beacon_names, 'rssi': beacon_rssis, \
                                    'dist': np.full(n, np.nan), 'dist2d': np.full(n, np.nan), 'est': np.full(n, np.nan)})
    if n_signals is not None:
        beacon_data.sort_values(by='rssi', ascending=False, ignore_index=True, inplace=True)
        beacon_data.drop(beacon_data.index[n_signals:], inplace=True)

    # create measurement
    measurement = vc.Measurement(device_name, beacon_data)
    if keep_actual_location:
        measurement.actual_location = device_location
    return measurement

# load data
beacons = pd.read_csv("beacons_proj.csv")
# create list of measurements
device_name = "d1"
n_signals = 8
anzahl_positionen=20
anzahl_devices=2
measurements = []
for i in range(0,anzahl_positionen):
    x, y = np.random.uniform(low=10, high=90, size=2)
    # x = round(x, 2)
    # y = round(y, 2)
    # print(x,y)
    meas = create_measurement(device_name, np.array([x,y,0.0]), beacons, n_signals, keep_dists=False, noise=0.5)
    measurements.append(meas)
    # print("meas x:{} y:{}".format(x,y))
    # print("meas \n{}".format(meas))
    if i % 2000 == 0:
        print("i: {}, {} %".format(i, i/anzahl_positionen*100))


# store measurements as pickle file
filename = "measurement1_test.p"
with open(filename, 'wb') as f:
    pickle.dump(measurements, f)

## TODO für Testdaten
# - zusammenhang zwischen stationen und coordinaten muss gegeben sein,
#   - wenn ein Datensatz an x:15 y:5 als stärkste station b2 hat muss das in der umgebung dieser position immer so sein
#