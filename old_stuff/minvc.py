# Visual Computing: Localization by Multilateration
# Utility classes
# Volker Ahlers, Hochschule Hannover, 2021

import numpy as np
import pandas as pd


# Beacon class storing name and 3D location
# (currently unused)
class Beacon:

    def __init__(self, name, location):
        assert type(name) is str
        self.name = name
        assert type(location) is np.ndarray
        assert location.shape == (3,)
        self.location = location

    def __repr__(self):
        return "beacon: " + self.name + ", location: " + str(self.location)


# Measurement class storing a measurement
# with timestamp and optional actual device location (for training)
class Measurement:
    
    def __init__(self, device_name, beacon_data, \
                 timestamp=pd.Timestamp.now(), actual_location=None):
        # name/id of the measurement device
        assert type(device_name) is str
        self.device_name = device_name
        # DataFrame with columns: name, rssi, optional dist
        assert type(beacon_data) is pd.DataFrame
        self.beacon_data: pd.DataFrame = beacon_data
        # date and time of measurement
        assert type(timestamp) is pd.Timestamp
        self.timestamp = timestamp
        # optional actual device location (for training)
        if actual_location is not None:
            assert type(actual_location) is np.ndarray
            assert actual_location.shape == (3,)
        self.actual_location = actual_location
        #calculated position
        self.device_est_position = None
        #uncertainties vector (standard deviation)
        self.uncertainties = None
        
    def __repr__(self):
        return "device: " + self.device_name + ", timestamp: " + str(self.timestamp) \
               + ", est_location: " + str(self.device_est_position) \
               + ", uncertainties: "+str(self.uncertainties) \
               +"\n" + str(self.beacon_data)

    #getter and setter
    def get_device_est_position(self):
        return self.device_est_position
    def set_device_est_position(self, device_est_position):
        self.device_est_position = device_est_position
    def get_uncertainties(self):
        return self.uncertainties
    def set_uncertainties(self, uncertainties):
        self.uncertainties = uncertainties

    #get calculated position as np.array
    def get_beacon_est(self):
        return self.beacon_data['est'].values
    # set calculated position as np.array
    def set_beacon_est(self, list_est):
        self.beacon_data['est'] = list_est

    # get beacon names as np.array
    def get_beacon_names(self):
        return self.beacon_data['name'].values

    # get beacon RSSI values as np.array
    def get_beacon_rssis(self):
        return self.beacon_data['rssi'].values

    # # get beacon distances as np.array
    # def get_beacon_dists(self):
    #     return self.beacon_data['dist'].values

    # def get_real_location(self):
    #     return self.actual_location

    # def set_real_location(self, real_location):
    #    self.actual_location=real_location

    # get 2D beacon distances as np.array
    # def get_beacon_dists2d(self):
    #    return self.beacon_data['dist2d'].values

    
# utility class to convert distance to RSSI and vice versa
class RSSIConverter:

    def __init__(self, c0=-30.0, n=2.0, d0=1.0):
        # RSSI value at reference distance d0
        self.c0 = c0
        # path-loss exponent (signal attenuation)
        self.n = n
        # reference distance d0
        self.d0 = d0
        
    # convert distance to RSSI
    def get_rssi(self, dist):
        return self.c0 - 10.0 * self.n * np.log10(dist / self.d0)
    
    # convert RSSI to distance
    def get_dist(self, rssi):
        return self.d0 * 10**((self.c0 - rssi) / (10.0 * self.n))
    