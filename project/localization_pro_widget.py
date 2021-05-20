# Visual Computing: Localization by Multilateration
# Localization of device from beacon RSSI measurement
# Volker Ahlers, Hochschule Hannover, 2021

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats
import minvc as vc

import matplotlib
#import matplotlib.animation as animation
from matplotlib.widgets import Slider


debug_bool = False
# read beacon locations
def load_beacon_locations() -> vc.Beacon:
    beacons: vc.Beacon = pd.read_csv("beacons_proj.csv", header=0, index_col="name")
    if debug_bool:
        print(beacons)

    return beacons

# read meas
def load_measurements():
    filename = "measurement_proj.p"
    with open(filename, 'rb') as f:
        measurements = pickle.load(f)
    if debug_bool:
        print(type(measurements))
        print(measurements)

    print("#meas: {}".format(len(measurements)))
    meas = []
    names_set=set()
    timestamps_set=set()
    mean_beacon_per_meas=0
    for measurement in measurements:
        measurement.beacon_data.drop('dist', 1, inplace=True)
        measurement.beacon_data.drop('dist2d', 1, inplace=True)

        meas.append(measurement)
        names_set.add(measurement.device_name)
        timestamps_set.add(measurement.timestamp)
        mean_beacon_per_meas+=len(measurement.get_beacon_names())

    mean_beacon_per_meas/=len(measurements)
    print("#beacons per meas (avg. ): {}".format(mean_beacon_per_meas))
    print("devices: {}".format(names_set))
    print("timestamps: {}".format(timestamps_set))
    print("#timestamps: {}".format(len(timestamps_set)))

    # select one measurement
    #meas: vc.Measurement = measurements[0]
    #if p:print(meas)
    return meas

def load_calibration():
    calib = pd.read_csv("calibration_proj.csv")
    if debug_bool:print(calib)
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
    fig = plt.figure(figsize=[12,12])
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

    beacons_location_mean=get_mean(beacons, meas)
    ax.add_patch(plt.Circle(beacons_location_mean[0:2], radius=location_dot, color='black', fc='black'))
    plt.annotate("mean", beacons_location_mean[0:2] + offset)
    add_if_not(legend, "mean")
    cir_list = []
    for name in beacons.index.values.tolist():
        beacon_location = beacons.loc[name].values
        cir = plt.Circle(beacon_location[0:2], radius=b_dot, fc='b')
        cir_list.append(cir)
        ax.add_patch(cir)

    plt.legend(handles=cir_list)

    for name, est in zip(meas.get_beacon_names(), meas.get_beacon_est()):
        beacon_location = beacons.loc[name].values

        if ~np.isnan(est):
            disc = "Estimated distance"
            # print("add {}".format(disc))
            ax.add_patch(plt.Circle(beacon_location[0:2], radius=est, alpha=0.3, color='red', fill=False))
            add_if_not(legend,disc)
            beacons_annotation = "{} est: {:.2f}".format(name, est)
            plt.annotate(beacons_annotation, beacon_location[0:2] + offset)

    plt.axis([0, 100, 0, 100])
    plt.legend(legend)
    plt.show()


# scatter plot of RSSI vs. distance
def visualize_rssi_dist(calib, c0, n):
    rssis= calib['rssi'].values
    dists = calib['dist'].values

    #calc linear dependence
    res = stats.linregress(dists, rssis)
    #print("intercept: {}, slope: {}".format(res.intercept,res.slope))
    #print("RSSIs {}".format(rssis))
    #plot rssis
    plt.scatter(dists, rssis, c='b', label='RSSIs')
    #plot line
    plt.plot(dists, res.intercept + res.slope*dists, 'g', label='fitted line')

    #plot fitted curve
    min=np.amin(dists)
    max=np.amax(dists)
    d = np.linspace(min, max, 20)
    plt.plot(d, c0 - 10.0 * n * np.log10(d), 'r', label='fitted curve')

    plt.title("Dependence of RSSI on distance")
    plt.xlabel("Distance")
    plt.ylabel("RSSI")
    plt.legend()
    plt.show()

def calc_dists_with_calibs(meas, c0, n):
    rssi_conv = vc.RSSIConverter(c0=c0, n=n, d0=1.0)
    beacon_ests = rssi_conv.get_dist(meas.get_beacon_rssis())
    meas.set_beacon_est(beacon_ests)
    return meas

def calc_c0_n(calib):
    rssis = calib['rssi'].values
    dists = calib['dist'].values

    #put dists and rssis in linreg to calc c0 and n
    # from log-distance path loss model for calibration
    res = stats.linregress(-10.0 * np.log10(dists), rssis)

    #first try static data
    #c0 = -50
    #n = 2

    #calculated data
    #c0: -37.45877267729592
    #n: 2.689458542605622
    c0=res.intercept
    n=res.slope
    print("Calibration result: c0: {}, n: {}".format(c0, n))

    #print("RSSIs {}".format(rssis))

    #for dist, rssi in zip(dists, rssis):
    #    print("dist {}, rssi {} rssi_est {}".format(dist, rssi, res.intercept + res.slope * dist))
    #    print("rssi {} dist {}, dist_est {}".format(rssi, dist, (rssi - res.intercept) / res.slope))
    # rssi = inter + slopw * dist
    # r = i+s*d

    return c0, n

def get_mean(beacons, meas):
    beacons_locations=beacons.loc[meas.get_beacon_names()].values
    #np.mean(beacons[1:4], 0)
    mean=beacons_locations.mean(axis=0)
    return mean

def residual(device_location, beacon_locations, measured_dists):
    """'device_location' is estimated tracking device location"""
    #delta_i=d_i-d(r_i,p)
    diff_ri_p=beacon_locations-device_location
    dist_ri_p=np.linalg.norm(diff_ri_p,axis=1)
    cur_error=measured_dists-dist_ri_p
    return cur_error

def calc_location(beacons, meas):
    initial_location=get_mean(beacons, meas)
    #print("mean: {}".format(initial_location))

    beacon_locations = beacons.loc[meas.get_beacon_names()].values
    #beacon_dists=meas.get_beacon_dists()
    beacon_dists=meas.get_beacon_est()

    #running least squares
    result=optimize.least_squares(residual, initial_location, args=(beacon_locations, beacon_dists))
    solution=result.x
    #print("solution: {}".format(solution))

    #meas.set_device_est_position([10,45]) # <- zum testen
    meas.set_device_est_position(solution)
    #meas.set_real_location(solution)
    return meas

def visualize_device_in_time_update(measurements, beacons, timestamp_index, ax):

    #TODO: need to be sure, that there is only one measuremnt for timestamp
    meas=measurements[timestamp_index]

    #prepare axes
    ax.clear()
    ax.axis([0, 100, 0, 100])

    col_est='red'
    col_beacon='blue'
    col_mean='black'
    col_dist='red'

    b_dot = 0.25                    # radius for dots
    location_dot = 0.45
    offset = np.array([0.6,0.3])    # offset for annotations
    offset_dist = np.array([0.6,-1])    # offset for annotations
    legend: list = []

    #estimated location
    pos = meas.get_device_est_position()
    if pos is not None:
        disc = "Estimated location"
        ax.add_patch(plt.Circle(pos[0:2], radius=location_dot, fc=col_est))
        ax.annotate(disc, pos[0:2] + offset)
        add_if_not(legend, disc)

    #mean
    beacons_location_mean=get_mean(beacons, meas)
    if beacons_location_mean is not None:
        disc = "Mean"
        ax.add_patch(plt.Circle(beacons_location_mean[0:2], radius=location_dot, fc=col_mean))
        ax.annotate(disc, beacons_location_mean[0:2] + offset)
        add_if_not(legend, disc)

    #beacons
    cir_list = []
    for name in beacons.index.values.tolist():
        beacon_location = beacons.loc[name].values
        cir = plt.Circle(beacon_location[0:2], radius=b_dot, fc=col_beacon, alpha=0.3)
        cir_list.append(cir)
        ax.add_patch(cir)
        ax.annotate(name, beacon_location[0:2] + offset, alpha=0.3)
    ax.legend(handles=cir_list)

    #distances
    for name, est in zip(meas.get_beacon_names(), meas.get_beacon_est()):
        beacon_location = beacons.loc[name].values
        if ~np.isnan(est):
            disc = "Estimated distance"
            # print("add {}".format(disc))
            def clamp(n, smallest, largest): return max(smallest, min(n, largest))#from: https://stackoverflow.com/questions/4092528/how-to-clamp-an-integer-to-some-range
            alpha=clamp(abs(1.0/100*(est-50)),0,1)
            #print(alpha)
            ax.add_patch(plt.Circle(beacon_location[0:2], radius=est, alpha=alpha, color=col_dist, fill=False))
            ax.add_patch(plt.Circle(beacon_location[0:2], radius=est, alpha=alpha/10, color=col_dist, fill=True))
            add_if_not(legend, disc)
            #beacons_annotation = "{} est: {:.2f}".format(name, est)
            #ax.annotate(name, beacon_location[0:2] + offset)
            ax.annotate("est: {:.2f}".format(est), beacon_location[0:2] + offset_dist)

            #emph measured beacons
            ax.annotate(name, beacon_location[0:2] + offset)
            ax.add_patch(plt.Circle(beacon_location[0:2], radius=b_dot, fc=col_beacon))


    #TODO: trace route for multiple devices
    if timestamp_index>0:
        for i in range(1,timestamp_index+1):
            from_loc=measurements[i-1].get_device_est_position()[0:2]
            to_loc=measurements[i].get_device_est_position()[0:2]
            #print("from: {} to: {}".format(from_loc,to_loc))
            ax.plot([from_loc[0], to_loc[0]],[from_loc[1], to_loc[1]], 'bo-')
            ax.annotate("time: {}".format(measurements[i-1].timestamp), from_loc + offset_dist)
            
    ax.legend(legend, loc="lower right")

def main():
    # vis calib
    calib = load_calibration()
    c0, n = calc_c0_n(calib)
    #visualize_rssi_dist(calib, c0, n)

    #load measured data
    beacons = load_beacon_locations()
    #ONLY FIRST ONE FOR START...
    meas = load_measurements()

    #calc dists for loaded data
    name = []
    time = []
    location = []
    for measurement in meas:
        #calc dists to beacons from rssi
        measurement = calc_dists_with_calibs(measurement, c0, n)

        #calc position of this measurement (device and time)
        measurement = calc_location(beacons, measurement)

        name.append(measurement.device_name)
        time.append(measurement.timestamp)
        location.append(measurement.device_est_position)

        #vis device
        #visualize_device(measurement, beacons)

    #TODO Sort meas for timestamp (if there would be more ;) )

    #prepare plot
    fig, ax = plt.subplots(figsize=[12, 12])
    #init vis
    visualize_device_in_time_update(meas, beacons, 0, ax)
    #Slider (widget example adapted from: https://riptutorial.com/matplotlib/example/23577/interactive-controls-with-matplotlib-widgets)
    #slider axes
    slider_ax = plt.axes([0.25, .03, 0.50, 0.02])
    timestamp_slider = Slider(slider_ax, 'Timestamp', 0, len(time)-1, valinit=0, valstep=1)
    #defined locally to have all values here
    def update_vis(val):
        #get selected timestamp
        timestamp=time[val]
        print("timestamp changed ({})".format(timestamp))
        #rebuild vis on axes
        visualize_device_in_time_update(meas, beacons, val, ax)
        fig.canvas.draw_idle()

    # call update function on slider value change
    timestamp_slider.on_changed(update_vis)
    #finally show plot
    plt.show()

    #save calculated data
    # print("{}, {}, {}".format(len(name), len(time), len(location)))
    d = {'name': name, 'time': time, 'location': location}
    df = pd.DataFrame(data=d)
    df.to_csv('out/out.csv', index=False)
    #print(meas)


    #CR
    #TODO done est or dist? --> est, remove rest
    # cleanup load functions
    #TODO done calc dist2d or remove ;) -->remove
    #TODO done save meas into csv
    # device: d0, timestamp: 2021-05-11 11:40:00, est_location: [47.09392202 48.1984661   2.44212059]

    #KD
    #TODO cleanup code
    # amd comment
    #TODO use only strongest rssi (how much?)
    # threashold or best 5?
    #TODO done: make nice visualization (on timestamps or something...)
    #TODO: highlight relevant beacons
    #TODO: trace route

    #delayed
    #TODO generate testdata with simluation (with real location) to test results
    #TODO check if calculated distances match the visualization

if __name__ == "__main__":
    main()