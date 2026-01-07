# -*-coding:utf-8-*
# The code refers to https://github.com/nwojke/deep_sort

import numpy as np
import scipy.linalg


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x1, y1, x2, y2, vx1, vy1, vx2, vy2

    contains the bounding box left & top position (x1, y1), right & bottom position (x2, y2)

    Object motion follows a constant velocity model. The bounding box location
    (x1, y1, x2, y2) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        # Create Kalman filter model matrices.
        # F
        self._motion_mat = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        # H
        self._update_mat = np.array(  # _update_mat
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )
        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x1, y1, x2, y2) with left & top position (x1, y1)
            , right & bottom position (x2, y2)

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * (measurement[2] - measurement[0]),
            2 * self._std_weight_position * (measurement[3] - measurement[1]),
            2 * self._std_weight_position * (measurement[2] - measurement[0]),
            2 * self._std_weight_position * (measurement[3] - measurement[1]),
            
            10 * self._std_weight_velocity * (measurement[2] - measurement[0]),
            10 * self._std_weight_velocity * (measurement[3] - measurement[1]),
            10 * self._std_weight_velocity * (measurement[2] - measurement[0]),
            10 * self._std_weight_velocity * (measurement[3] - measurement[1]),
        ]
        covariance = np.diag(np.square(std))
        # 返回的mean需要从(x1, y1, x2, y2)转换为(x, y, a, h)
        x1, y1, x2, y2 = mean[0], mean[1], mean[2], mean[3]
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        a = w / h if h != 0 else 0
        mean[0], mean[1], mean[2], mean[3] = x, y, a, h
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * (mean[2] - mean[0]),
            self._std_weight_position * (mean[3] - mean[1]),
            self._std_weight_position * (mean[2] - mean[0]),
            self._std_weight_position * (mean[3] - mean[1]),
        ]
        std_vel = [
            self._std_weight_velocity * (mean[2] - mean[0]),
            self._std_weight_velocity * (mean[3] - mean[1]),
            self._std_weight_velocity * (mean[2] - mean[0]),
            self._std_weight_velocity * (mean[3] - mean[1]),
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        # 返回的mean需要从(x1, y1, x2, y2)转换为(x, y, a, h)
        x1, y1, x2, y2 = mean[0], mean[1], mean[2], mean[3]
        w = x2 - x1
        h = y2 - y1
        if w <= 0:
            w = 1e-4  # 给予一个极小的正数
            x2 = x1 + w
        if h <= 0:
            h = 1e-4
            y2 = y1 + h
        x = x1 + w / 2
        y = y1 + h / 2
        a = w / h
        mean[0], mean[1], mean[2], mean[3] = x, y, a, h
        return mean, covariance


    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * (mean[2] - mean[0]),
            self._std_weight_position * (mean[3] - mean[1]),
            self._std_weight_position * (mean[2] - mean[0]),
            self._std_weight_position * (mean[3] - mean[1]),
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x1, y1, x2, y2)
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        # Hx 和 (HPH^T+R)
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        # K
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        # y = z - Hx
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        # 返回的mean需要从(x1, y1, x2, y2)转换为(x, y, a, h)
        x1, y1, x2, y2 = new_mean[0], new_mean[1], new_mean[2], new_mean[3]
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        a = w / h if h != 0 else 0
        new_mean[0], new_mean[1], new_mean[2], new_mean[3] = x, y, a, h
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:4], covariance[:4, :4]
            measurements = measurements[:, :4]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')