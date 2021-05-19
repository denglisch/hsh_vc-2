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
# beacons: vc.Beacon = pd.read_csv("beacons.csv")
beacons: vc.Beacon = pd.read_csv("beacons_proj.csv")
beacons.set_index('name', inplace=True)
# display(beacons)
print(beacons)

# read measurements
# filename = "project/measurement_proj.p"
# with open(filename, 'rb') as f:
#     measurements = pickle.load(f)
# print(type(measurements))
# print(measurements)

# filename = "measurement1noise3.p"
filename = "measurement_proj.p"
with open(filename, 'rb') as f:
    measurements = pickle.load(f)
print(type(measurements))


# select one measurement
meas: vc.Measurement = measurements[0]
print(meas)


# visualize beacon locations
def visualize(meas, beacons, c0, n):
    r_dot = 0.25                  # radius for dots
    offset = np.array([0.6,0.3])  # offset for annotations

    fig = plt.figure(figsize=[8,8])
    ax = plt.axes()
    # ax.add_patch(plt.Circle(meas.get_real_location()[0:2], radius=0.10, fc='red'))
    # plt.annotate("real", meas.get_real_location()[0:2] + offset)

    beacon_names = meas.get_beacon_names()
    beacon_dists = meas.get_beacon_dists()
    beacon_dists2d = meas.get_beacon_dists2d()

    # rssi_conv = vc.RSSIConverter( c0=c0, n=n, d0=1.0)
    rssi_conv = vc.RSSIConverter()
    beacon_estimated = rssi_conv.get_dist(meas.get_beacon_rssis())

    print("beacon names: {} {} {}".format(beacon_names, beacon_dists, beacon_estimated))
    for name, dist, estimated, dist2d in zip(beacon_names, beacon_dists, beacon_estimated, beacon_dists2d):
        beacon_location = beacons.loc[name].values

        print("beacon: {} x: {} y:{} ".format(name, beacon_location[0], beacon_location[1]))
        # [0:2] nur index 0 bis exkl 2 nehmen
        ax.add_patch(plt.Circle(beacon_location[0:2], radius=r_dot, fc='b'))
        # ax.add_patch(plt.Circle(beacon_location[0:2], radius=dist, alpha=0.3, color='red', fill=False))
        ax.add_patch(plt.Circle(beacon_location[0:2], radius=estimated, alpha=0.3, color='orange', fill=False))
        #ax.add_patch(plt.Circle(beacon_location[0:2], radius=dist2d, alpha=0.3, color='black', fill=False))

        plt.annotate("{} dist: {:.2f}".format(name ,dist), beacon_location[0:2] + offset)
    plt.axis('scaled')   # alternative: plt.axis([xmin, xmax, ymin, ymax])
    plt.legend(["real point", "beacon", "real dist", "estimated dist"])
    plt.show()

# beacon_locations = beacons.loc[meas.get_beacon_names()].values
# print("location")
# print(beacon_locations)

rssi_conv = vc.RSSIConverter()
beacon_estimated = rssi_conv.get_dist(meas.get_beacon_rssis())

res = stats.linregress(beacon_estimated, meas.get_beacon_rssis())
print("intercept: {}, slope: {}".format(res.intercept,res.slope))
print("RSSIs {}".format(meas.get_beacon_rssis()))



# scatter plot of RSSI vs. distance
plt.scatter(beacon_estimated, meas.get_beacon_rssis())
plt.plot(beacon_estimated, res.intercept + res.slope*beacon_estimated, 'r', label='fitted line')

plt.title("RSSI vs. distance")
plt.xlabel("Distance")
plt.ylabel("RSSI")
plt.legend()
plt.show()


visualize(meas, beacons, res.intercept, res.slope)