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
def load_beacon() -> vc.Beacon:
    beacons: vc.Beacon = pd.read_csv("beacons_proj.csv")
    beacons.set_index('name', inplace=True)
    print(beacons)
    return beacons

# read meas
def load_meas():
    filename = "measurement_proj.p"
    with open(filename, 'rb') as f:
        measurements = pickle.load(f)
    print(type(measurements))
    # select one measurement
    meas: vc.Measurement = measurements[0]
    print(meas)
    return meas

def load_calibration():
    calib = pd.read_csv("calibration_proj.csv")
    print(calib)
    return calib

def add_if_not(liste, value):
    if value not in liste:
        liste.append(value)

# visualize beacon locations
def visualize_device(meas, beacons):
    b_dot = 0.25                    # radius for dots
    location_dot = 0.45
    offset = np.array([0.6,0.3])    # offset for annotations
    legend: list = []
    fig = plt.figure(figsize=[8,8])
    ax = plt.axes()

    real_pos = meas.get_real_location()
    if real_pos is not None:
        disc = "real location"
        print("add {}".format(disc))
        ax.add_patch(plt.Circle(real_pos[0:2], radius=location_dot, fc='orange'))
        # plt.annotate(disc, real_pos[0:2] + offset)
        add_if_not(legend, disc)

    pos = meas.get_device_est_position()
    if pos is not None:
        disc = "Estimated location"
        print("add {}".format(disc))

        ax.add_patch(plt.Circle(pos[0:2], radius=location_dot, fc='red'))
        # plt.annotate(disc, pos[0:2] + offset)
        add_if_not(legend, disc)

    for name, dist, est in zip(meas.get_beacon_names(), meas.get_beacon_dists(), meas.get_beacon_est()):
        beacon_location = beacons.loc[name].values

        print("beacon: {} x: {} y: {} dist: {} est: {} ".format(name, beacon_location[0], beacon_location[1], dist, est))
        # [0:2] nur index 0 bis exkl 2 nehmen
        ax.add_patch(plt.Circle(beacon_location[0:2], radius=b_dot, fc='b'))
        add_if_not(legend,"Beacon location")
        beacons_annotation = "{}".format(name)

        if ~np.isnan(dist):
            print("add dist")
            ax.add_patch(plt.Circle(beacon_location[0:2], radius=dist, alpha=0.3, color='orange', fill=False))
            add_if_not(legend,"real Dist")
            beacons_annotation = "{} dist: {:.2f}".format(beacons_annotation, dist)

        if ~np.isnan(est):
            disc = "Estimated distance"
            # print("add {}".format(disc))
            ax.add_patch(plt.Circle(beacon_location[0:2], radius=est, alpha=0.3, color='red', fill=False))
            add_if_not(legend,disc)
            beacons_annotation = "{} est: {:.2f}".format(beacons_annotation, est)
        #ax.add_patch(plt.Circle(beacon_location[0:2], radius=dist2d, alpha=0.3, color='black', fill=False))
        plt.annotate(beacons_annotation, beacon_location[0:2] + offset)


    plt.axis('scaled')   # alternative: plt.axis([xmin, xmax, ymin, ymax])
    plt.legend(legend)
    plt.show()


# scatter plot of RSSI vs. distance
def visualize_rssi_dist(calib):
    rssi = calib['rssi'].values
    dist = calib['dist'].values
    res = stats.linregress(dist, rssi)
    print("intercept: {}, slope: {}".format(res.intercept,res.slope))
    print("RSSIs {}".format(rssi))
    plt.scatter(dist, rssi)
    plt.plot(dist, res.intercept + res.slope*dist, 'r', label='fitted line')

    plt.title("RSSI vs. distance")
    plt.xlabel("Distance")
    plt.ylabel("RSSI")
    plt.legend()
    plt.show()

def calc_est(meas, c0, n):
    rssi_conv = vc.RSSIConverter(c0=c0, n=n, d0=1.0)
    beacon_est = rssi_conv.get_dist(meas.get_beacon_rssis())
    meas.set_beacon_est(beacon_est)
    return meas

def calc_c0_n(calib):
    # ??? hier muss c0 und n berechnet werden
    c0 = -50
    n = 2
    return c0, n

def calc_location(meas):
    # ??? Hier fehtl die berechnung des punktes
    meas.set_device_est_position([10,45]) # <- zum testen
    return meas

def main():
    # vis calib
    calib = load_calibration()
    visualize_rssi_dist(calib)
    c0, n = calc_c0_n(calib)

    #load
    beacons = load_beacon()
    meas = load_meas()

    #calc
    meas = calc_est(meas, c0, n)
    meas = calc_location(meas)

    #vis device
    visualize_device(meas, beacons)


if __name__ == "__main__":
    main()