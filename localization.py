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
beacons: vc.Beacon = pd.read_csv("beacons.csv")
beacons.set_index('name', inplace=True)
# display(beacons)
print(beacons)

# read measurements
# filename = "project/measurement_proj.p"
# with open(filename, 'rb') as f:
#     measurements = pickle.load(f)
# print(type(measurements))
# print(measurements)

filename = "measurement1.p"
with open(filename, 'rb') as f:
    measurements = pickle.load(f)
print(type(measurements))


# select one measurement
meas: vc.Measurement = measurements[0]
print(meas)


# visualize beacon locations
def visualize(meas, beacons):
    r_dot = 0.25                  # radius for dots
    offset = np.array([0.6,0.3])  # offset for annotations

    fig = plt.figure(figsize=[8,8])
    ax = plt.axes()
    ax.add_patch(plt.Circle(meas.get_real_location()[0:2], radius=0.10, fc='red'))
    plt.annotate("real", meas.get_real_location()[0:2] + offset)

    beacon_names = meas.get_beacon_names()
    beacon_dists = meas.get_beacon_dists()
    beacon_dists2d = meas.get_beacon_dists2d()

    rssi_conv = vc.RSSIConverter( c0=-29.0, n=2.3, d0=1.0)
    beacon_rssis = rssi_conv.get_dist(meas.get_beacon_rssis())

    print("beacon names: {}".format(beacon_names))
    for name, dist, dist2d, rssi in zip(beacon_names, beacon_dists, beacon_dists2d, beacon_rssis):
        beacon_location = beacons.loc[name].values

        print("beacon: {} x: {} y:{} ".format(name, beacon_location[0], beacon_location[1]))
        # [0:2] nur index 0 bis exkl 2 nehmen
        ax.add_patch(plt.Circle(beacon_location[0:2], radius=rssi, alpha=0.3, color='g', fill=False))
        ax.add_patch(plt.Circle(beacon_location[0:2], radius=dist, alpha=0.3, color='orange', fill=False))
        ax.add_patch(plt.Circle(beacon_location[0:2], radius=r_dot, fc='b'))
        plt.annotate("{} dist: {:.2f}".format(name ,dist), beacon_location[0:2] + offset)
    plt.axis('scaled')   # alternative: plt.axis([xmin, xmax, ymin, ymax])
    plt.legend(["real", "dist2d", "dist", "beacon"])
    plt.show()

# beacon_locations = beacons.loc[meas.get_beacon_names()].values
# print("location")
# print(beacon_locations)


# visualize(meas, beacons)
res = stats.linregress(meas.get_beacon_dists(), meas.get_beacon_rssis())
print(res)
print(meas.get_beacon_rssis())

# scatter plot of RSSI vs. distance
plt.scatter(meas.get_beacon_dists(), meas.get_beacon_rssis())
plt.plot(meas.get_beacon_dists(), res.intercept + res.slope*meas.get_beacon_dists(), 'r', label='fitted line')
plt.legend()
plt.show()
