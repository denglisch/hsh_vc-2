# Visual Computing: Localization by Multilateration
# Localization of device from beacon RSSI measurement
# Volker Ahlers, Hochschule Hannover, 2021

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats
import minvc as vc

# read beacon locations
beacons = pd.read_csv("beacons.csv")
beacons.set_index('name', inplace=True)
#display(beacons)
print(beacons)

# read measurements
filename = "measurement1.p"
with open(filename, 'rb') as f:
    measurements = pickle.load(f)
print(type(measurements))
print(measurements)

# select one measurement
meas = measurements[0]
print(meas)

# visualize beacon locations
def visualize(meas, beacons):
    r_dot = 0.25                  # radius for dots
    offset = np.array([0.3,0.0])  # offset for annotations

    fig = plt.figure(figsize=[8,8])
    ax = plt.axes()
    beacon_names = meas.get_beacon_names()
    for name in beacon_names:
        beacon_location = beacons.loc[name].values
        ax.add_patch(plt.Circle(beacon_location[0:2], radius=r_dot, fc='b'))
        plt.annotate(name, beacon_location[0:2] + offset)
    plt.axis('scaled')   # alternative: plt.axis([xmin, xmax, ymin, ymax])
    plt.show()

visualize(meas, beacons)

# scatter plot of RSSI vs. distance
plt.scatter(meas.get_beacon_dists(), meas.get_beacon_rssis())
plt.show()
