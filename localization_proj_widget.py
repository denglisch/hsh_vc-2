# Visual Computing: Localization by Multilateration
# Localization of device from beacon RSSI measurement
# Volker Ahlers, Hochschule Hannover, 2021

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats
import minvc as vc
import time as tm

from matplotlib.widgets import Slider
from matplotlib.patches import Ellipse
import tensorflow as tf

prediction = True
nan_values = 0
debug_bool = False
# read beacon locations
def load_beacon_locations() -> vc.Beacon:
    beacons: vc.Beacon = pd.read_csv("beacons_proj.csv", header=0, index_col="name")
    if debug_bool:
        print(beacons)

    return beacons

# read meas
def load_measurements():
    if prediction:
        filename = "measurement1_test.p"
    else:
        filename = "measurement_proj.p"
        # filename = "measurement1_test_old.p"
    with open(filename, 'rb') as f:
        measurements = pickle.load(f)
    if debug_bool:
        print(type(measurements))
        print(measurements)

    print("#meas: {}".format(len(measurements)))
    #lists for stats
    meas = []
    names_set=set()
    timestamps_set=set()
    mean_beacon_per_meas=0
    for measurement in measurements:
        #drop unused cols
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
    return meas

def load_calibration():
    calib = pd.read_csv("calibration_proj.csv")
    if debug_bool:print(calib)
    return calib

def add_if_not(list, value):
    """add value into list, if not already in"""
    if value not in list:
        list.append(value)

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
    #plt.plot(dists, res.intercept + res.slope*dists, 'g', label='fitted line')

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

def calc_dists_with_calibs(meas, c0, n, d0=1.0):
    """calculate distances from device to beacons in given measured data and writes them into meas-list.

    All measured data will be used. Currently there is no sort out.

    Parameter d0 is 1.0 by default"""
    rssi_conv = vc.RSSIConverter(c0=c0, n=n, d0=d0)
    beacon_ests = rssi_conv.get_dist(meas.get_beacon_rssis())
    meas.set_beacon_est(beacon_ests)
    return meas

def calc_c0_n(calib):
    """calculates values c0 and n according to calibration data by linear regression"""
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
    return c0, n

def get_mean(beacons, meas):
    """calculates the mean of beacons locations whose signal is within the measured data"""
    beacons_locations=beacons.loc[meas.get_beacon_names()].values
    mean=beacons_locations.mean(axis=0)
    return mean

def residual(device_location, beacon_locations, measured_dists):
    """'device_location' is estimated tracking device location according formula:

    delta_i=d_i-d(r_i,p)"""
    diff_ri_p=beacon_locations-device_location
    dist_ri_p=np.linalg.norm(diff_ri_p,axis=1)
    cur_error=measured_dists-dist_ri_p
    return cur_error

def calc_location(beacons, meas):
    """calculates the estimated location (est) as well as standard deviation of the device and updates values in given meas"""
    initial_location=get_mean(beacons, meas)
    #print("mean: {}".format(initial_location))

    beacon_locations = beacons.loc[meas.get_beacon_names()].values
    #beacon_dists=meas.get_beacon_dists()
    beacon_dists=meas.get_beacon_est()

    #running least squares
    result=optimize.least_squares(residual, initial_location, args=(beacon_locations, beacon_dists))
    solution=result.x

    #print("solution: {}".format(solution))
    meas.set_device_est_position(solution)

    #calc uncertainties
    #TODO: Assumption here: Jacobian is regular ;)
    # Implement singular-value decomposition
    J=result.jac
    H=J.T.dot(J)
    cov=np.linalg.inv(H)
    variance=np.diag(cov)
    sigma=np.sqrt(variance)
    #print("sigma: {}".format(sigma))
    meas.set_uncertainties(sigma)
    return meas


def visualize_device_in_time_update(measurements, beacons, timestamp_index, ax):
    """visualizes beacon locations within given axes (ax).

    This function should be called from visualize_device_2d(meas, beacons, time)
    """
    #TODO: need to be sure, that there is only one measuremnt for timestamp
    meas=measurements[timestamp_index]

    #prepare axes
    ax.clear()
    ax.axis([0, 100, 0, 100])

    col_pred = 'violet'
    col_est='red'
    col_beacon='blue'
    col_mean='black'
    col_dist='red'
    col_trace='gray'

    b_dot = 0.25                    # radius for dots
    location_dot = 0.45
    offset = np.array([0.6,0.3])    # offset for annotations
    offset_dist = np.array([0.6,-1])    # offset for annotations
    legend: list = []

    # predicted location
    if prediction:
        pos1 = meas.get_pred_location()[0]
        list = pos1.tolist()
        list.append(0)

        pos_pred = list

        if pos_pred is not None:
            uncert = [1, 1] * 5
            disc = "Real location"
            real_pos = meas.get_real_location()
            ellipse = Ellipse(real_pos[0:2], width=uncert[0], height=uncert[1], alpha=0.5, color=col_pred, fill=True)
            ax.add_patch(ellipse)
            ax.annotate(disc, real_pos[0:2] + offset)
            print("pred: {} real: {}".format(list, real_pos))
            disc = "Predicted location"
            ax.annotate(disc, pos_pred[0:2] + offset)
            add_if_not(legend, disc)
            ellipse = Ellipse(pos_pred[0:2], width=uncert[0], height=uncert[1], alpha=0.5, color=col_pred, fill=True)
            ax.add_patch(ellipse)

    # estimated location
    pos = meas.get_device_est_position()
    if pos is not None:
        disc = "Estimated location"
        #ax.add_patch(plt.Circle(pos[0:2], radius=location_dot, fc=col_est))
        ax.annotate(disc, pos[0:2] + offset)
        add_if_not(legend, disc)
        #multiply by 5 to see anything ;)
        uncert=meas.get_uncertainties()*5
        ellipse=Ellipse(pos[0:2], width=uncert[0], height=uncert[1], alpha=0.5, color=col_est, fill=True)
        print("Uncertainties sigma: {}".format(meas.get_uncertainties()))
        #print(ellipse)
        ax.add_patch(ellipse)

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
            ax.annotate("est: {:.2f}".format(est), beacon_location[0:2] + offset_dist)

            #emph measured beacons
            ax.annotate(name, beacon_location[0:2] + offset)
            ax.add_patch(plt.Circle(beacon_location[0:2], radius=b_dot, fc=col_beacon))


    #TODO: trace route for multiple devices
    if timestamp_index>0:
        label = "Trajectory"
        for i in range(1,timestamp_index+1):
            #two lines to repair legend ;)
            if i>1 :label="_nolegend_"
            else: legend.insert(0,label)
            from_loc=measurements[i-1].get_device_est_position()[0:2]
            to_loc=measurements[i].get_device_est_position()[0:2]
            #print("from: {} to: {}".format(from_loc,to_loc))
            ax.plot([from_loc[0], to_loc[0]],[from_loc[1], to_loc[1]], 'o', ms=2.0, ls='-', lw=1.0, color=col_trace, label=label)
            ax.annotate("time: {}".format(measurements[i-1].timestamp), from_loc + offset_dist)

    ax.legend(legend, loc="lower right")

def visualize_device_2d(meas, beacons, time):
    """
    Builds a mathplotlib widget with slider for timestamps.

    Calls visualize_device_in_time_update()
    """
    #prepare plot
    fig, ax = plt.subplots(figsize=[12, 12])
    #init vis
    visualize_device_in_time_update(meas, beacons, 0, ax)
    #Slider (widget example adapted from: https://riptutorial.com/matplotlib/example/23577/interactive-controls-with-matplotlib-widgets)
    #slider axes
    #slider_ax = plt.axes([0.25, .03, 0.50, 0.02])
    slider_ax = plt.axes([0.35, .03, 0.50, 0.02])
    timestamp_slider = Slider(slider_ax, "Timestamp: {}".format(time[0]), 0, len(time)-1, valinit=0, valstep=1)
    #defined locally to have all values here
    def update_vis(val):
        #get selected timestamp
        timestamp=time[val]
        print("Replot timestamp {}".format(timestamp))
        timestamp_slider.label.set_text("Timestamp: {}".format(timestamp))
        #rebuild vis on axes
        visualize_device_in_time_update(meas, beacons, val, ax)
        fig.canvas.draw_idle()

    # call update function on slider value change
    timestamp_slider.on_changed(update_vis)
    #finally show plot
    plt.show()

def save_to_csv(name, time, location, pred):
    """Saves calculated location data into csv file"""
    temp_df = pd.DataFrame(data=location, columns=['x', 'y','z'])
    if prediction:
        filenname = 'out/pred_test_{}.csv'.format(tm.strftime("%Y-%m-%d_%H-%M-%S"))
        d = {'name': name, 'time': time, 'location': location, 'predicted_loc': pred}
    else:
        filenname = 'out/distances.csv'
        d = {'name': name, 'time': time, 'x': temp_df['x'], 'y': temp_df['y'], 'z': temp_df['z']}

    df = pd.DataFrame(data=d)
    df.to_csv(filenname, index=False)

def predict_location(beacons, meas):
    """predict a location with AI learned by tensor.py"""
    # load Model
    model = tf.keras.models.load_model('saved_model/my_model.h5')
    # fill input list with nan_value=0
    row = np.full(50, nan_values).tolist()

    # add generatet data
    for name, rssi in zip(meas.get_beacon_names(), meas.get_beacon_rssis()):
        index = int(name.replace("b", ""))
        # norm
        row[index] = ((-rssi)-0)/(100-0)

    df = pd.DataFrame([row], columns=beacons.index.values.tolist())
    solution = model.predict(df)

    meas.set_pred_location(solution)
    meas.set_uncertainties(2)
    return meas


def main():
    # vis calib
    print("Load calibration data")
    calib = load_calibration()
    c0, n = calc_c0_n(calib)
    visualize_rssi_dist(calib, c0, n)

    #load measured data
    print("Load beacons data")
    beacons = load_beacon_locations()
    print("Load measured data")
    meas = load_measurements()

    #Sort meas for timestamp (if there would be more ;) )
    meas.sort(key=lambda m:m.timestamp)

    #calc dists for loaded data
    print("Calculate distances")
    name = []
    time = []
    location = []
    predicted_location = []
    for measurement in meas:
        #calc dists to beacons from rssi
        measurement = calc_dists_with_calibs(measurement, c0, n)

        if prediction:
            #  predict position of this measurement (device and time)
            measurement = predict_location(beacons, measurement)
            predicted_location.append(measurement.get_pred_location())
        #  calc position of this measurement (device and time)
        measurement = calc_location(beacons, measurement)

        name.append(measurement.device_name)
        time.append(measurement.timestamp)
        location.append(measurement.device_est_position)


    print(time)
    print("Visualize 2D")
    visualize_device_2d(meas, beacons, time)


    #save calculated data
    print("Save distances data")
    save_to_csv(name, time, location, predicted_location)


    #CR
    #TODO save locations as x, y, z
    # device: d0, timestamp: 2021-05-11 11:40:00, x: 47.09392202, y: 48.1984661, z: 2.44212059

    #Open
    #TODO: use only strongest rssi (how much?)
    # threashold or best 5?
    # Idea: Simulate data where real position is known, then check for best accurancy
    #TODO: highlight relevant beacons

if __name__ == "__main__":
    main()