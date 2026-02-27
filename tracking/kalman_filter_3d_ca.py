# Author: wangxy
# Implements Constant Acceleration (CA) model for 3D tracking

import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox3D):
        """
        Initialises a tracker using initial bounding box.
        State: [x, y, z, ry, l, w, h, vx, vy, vz, ax, ay, az] (13 dimensions)
        """
        self.dt = 0.1
        self.kf = KalmanFilter(dim_x=13, dim_z=7)
        
        # State Transition Matrix F
        # x = x + vt + 0.5at^2
        # v = v + at
        F = np.eye(13)
        # Position updates (x, y, z)
        F[0, 7] = self.dt; F[0, 10] = 0.5 * self.dt**2
        F[1, 8] = self.dt; F[1, 11] = 0.5 * self.dt**2
        F[2, 9] = self.dt; F[2, 12] = 0.5 * self.dt**2
        
        # Velocity updates (vx, vy, vz)
        F[7, 10] = self.dt
        F[8, 11] = self.dt
        F[9, 12] = self.dt
        
        self.kf.F = F

        # Measurement Function H
        # We measure [x, y, z, ry, l, w, h]
        self.kf.H = np.zeros((7, 13))
        for i in range(7):
            self.kf.H[i, i] = 1.0

        # Measurement Uncertainty R
        # self.kf.R[0:,0:] *= 1.0  # Can be tuned
        # self.kf.R[3, 3] *= 1.0   # Orientation

        # Initial State Covariance P
        self.kf.P *= 10.
        self.kf.P[7:10, 7:10] *= 1000. # High uncertainty for initial velocity
        self.kf.P[10:13, 10:13] *= 1000. # High uncertainty for initial acceleration

        # Process Noise Q
        self.kf.Q[7:, 7:] *= 0.01
        # Give higher process noise to acceleration (jerk)
        # self.kf.Q[10:13, 10:13] *= 0.1

        self.kf.x[:7] = bbox3D.reshape((7, 1))
        
        # [APN] Backup original Q
        self.original_Q = self.kf.Q.copy()

    def update(self, bbox3D):
        """
        Updates the state vector with observed bbox.
        """
        # Orientation correction
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        new_ry = bbox3D[3]
        if new_ry >= np.pi: new_ry -= np.pi * 2
        if new_ry < -np.pi: new_ry += np.pi * 2
        bbox3D[3] = new_ry

        predicted_ry = self.kf.x[3]
        if abs(new_ry - predicted_ry) > np.pi / 2.0 and abs(new_ry - predicted_ry) < np.pi * 3 / 2.0:
            bbox3D[3] += np.pi
            if bbox3D[3] > np.pi: bbox3D[3] -= np.pi * 2
            if bbox3D[3] < -np.pi: bbox3D[3] += np.pi * 2
        new_ry = bbox3D[3]
        
        if abs(new_ry - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_ry > 0:
                self.kf.x[3] += np.pi * 2
            else:
                self.kf.x[3] -= np.pi * 2

        self.kf.update(bbox3D)

        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

    def predict(self, apn_cfg=None):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        Supports APN (Adaptive Process Noise).
        """
        # [APN Logic]
        if apn_cfg and apn_cfg.get('use_apn_ctra', False): # Using the same config key for simplicity
            try:
                # Calculate acceleration magnitude
                ax, ay, az = self.kf.x[10], self.kf.x[11], self.kf.x[12]
                accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
                
                # CA model usually assumes no rotation rate (omega=0), so we rely on accel
                params = apn_cfg.get('apn_params', {})
                k_accel = params.get('maneuver_factor_accel', 0.5)
                
                maneuver_factor = 1.0 + k_accel * float(accel_mag)
                self.kf.Q = self.original_Q * maneuver_factor
            except:
                pass

        self.kf.predict()
        
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
        
        # [APN] Restore Q
        if apn_cfg and apn_cfg.get('use_apn_ctra', False):
            self.kf.Q = self.original_Q

        return self.kf.x[:7].reshape((7,))

    def get_state(self):
        return self.kf.x[:7].reshape((7,))