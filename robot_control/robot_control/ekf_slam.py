"""
EKF SLAM for CSE 276A HW3 (landmark = AprilTag center on the floor plane).

Plain numpy, no ROS, so it can be tested and re-run on a laptop.
Notation follows the lecture slides (Lec 8a/8b, Lec 9):

    state      s = [x_r, z_1, ..., z_n]
                 x_r = (x, y, theta)  robot pose
                 z_j = (x_j, y_j)     landmark j position
    covariance Sigma (3 + 2n square)
    odometry   d = (ds, dtheta)       distance driven, heading change

    Prediction (Lec 9 odometry model, Lec 8a EKF):
        s_{i|i-1}     = f(s_{i-1|i-1}, d_i)
        Sigma_{i|i-1} = F Sigma F^T + W Q W^T
    Update with a sighting f_i of a KNOWN landmark:
        S     = H Sigma H^T + R
        K     = Sigma H^T S^-1
        s     = s + K (f_i - h(s))
        Sigma = (I - K H) Sigma          (Joseph form used for stability)
    Matching: tag IDs give the association; the Mahalanobis distance
        M = nu^T S^-1 nu is used as a validation gate (M < gamma) to drop
        bad detections.
    New landmark: augment the state vector and set up its covariance with
        the robot and the other features ("Strategy 2" on the Lec 8b slide).

Measurement model: range and bearing to the tag center, in the robot frame
    f = (r, b),  r = |z_j - p|,  b = atan2(z_j - p) - theta
The tag's tilt is not used: the center is measured much more reliably.

The filter does not know how many landmarks there are; the state grows by
two each time a new tag ID is seen.
"""
import math

import numpy as np


def wrap(a):
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


class EKFSlam:
    def __init__(self, start_pose=(0.0, 0.0, 0.0), start_sigma=(0.0, 0.0, 0.0),
                 k_s=0.05, k_theta=0.10, k_s_theta=0.05, min_sigma_s=0.005,
                 min_sigma_theta=0.005, sigma_r0=0.02, k_r=0.03,
                 sigma_b=math.radians(2.0), gate=9.21):
        """
        start_pose:  robot pose at t=0. The robot doesn't know where it is in
                     the room, so (0, 0, 0) defines the map frame.
        start_sigma: std devs of the start pose (0 = this pose defines the frame).
        Odometry noise Q (std devs, per motion):
            sigma_ds     = min_sigma_s + k_s * |ds|
            sigma_dtheta = min_sigma_theta + k_theta * |dtheta| + k_s_theta * |ds|
        Measurement noise R (std devs):
            sigma_r = sigma_r0 + k_r * r      (range error grows with distance)
            sigma_b = constant bearing error
        gate: Mahalanobis validation gate. 9.21 = chi^2 with 2 dof at 99%.
        """
        self.s = np.array(start_pose, dtype=float)
        self.Sigma = np.diag(np.square(np.asarray(start_sigma, dtype=float)))
        self.ids = []          # landmark IDs, in state-vector order
        self.k_s, self.k_theta, self.k_s_theta = k_s, k_theta, k_s_theta
        self.min_sigma_s, self.min_sigma_theta = min_sigma_s, min_sigma_theta
        self.sigma_r0, self.k_r, self.sigma_b = sigma_r0, k_r, sigma_b
        self.gate = gate

    # ---------------- helpers ----------------
    @property
    def pose(self):
        return float(self.s[0]), float(self.s[1]), float(self.s[2])

    @property
    def n_landmarks(self):
        return len(self.ids)

    def _idx(self, tag_id):
        """Index of landmark tag_id's x in the state vector."""
        return 3 + 2 * self.ids.index(tag_id)

    def landmarks(self):
        """{id: (x, y, 2x2 covariance)}"""
        out = {}
        for tid in self.ids:
            j = self._idx(tid)
            out[tid] = (float(self.s[j]), float(self.s[j + 1]),
                        self.Sigma[j:j + 2, j:j + 2].copy())
        return out

    def R_for(self, r):
        sr = self.sigma_r0 + self.k_r * r
        return np.diag([sr ** 2, self.sigma_b ** 2])

    # ---------------- prediction ----------------
    def predict(self, ds, dtheta):
        """
        Odometry step: drive ds meters while the heading changes by dtheta.
        A pure turn is predict(0, dtheta). Assumes the heading changes evenly
        during the motion (same model as Lec 9, slide 24).
        Only the robot part of the state moves; landmarks are static.
        """
        x, y, th = self.s[:3]
        thm = th + dtheta / 2.0
        c, sn = math.cos(thm), math.sin(thm)
        self.s[0] = x + ds * c
        self.s[1] = y + ds * sn
        self.s[2] = wrap(th + dtheta)

        # Jacobians: F wrt the robot pose, W wrt the odometry noise
        F = np.array([[1.0, 0.0, -ds * sn],
                      [0.0, 1.0, ds * c],
                      [0.0, 0.0, 1.0]])
        W = np.array([[c, -0.5 * ds * sn],
                      [sn, 0.5 * ds * c],
                      [0.0, 1.0]])
        sig_s = self.min_sigma_s + self.k_s * abs(ds)
        sig_t = (self.min_sigma_theta + self.k_theta * abs(dtheta)
                 + self.k_s_theta * abs(ds))
        Q = np.diag([sig_s ** 2, sig_t ** 2])

        S = self.Sigma
        S[:3, :3] = F @ S[:3, :3] @ F.T + W @ Q @ W.T
        S[:3, 3:] = F @ S[:3, 3:]
        S[3:, :3] = S[:3, 3:].T

    # ---------------- measurement model ----------------
    def h(self, j):
        """Predicted (r, b) of the landmark at state index j, and its Jacobian H."""
        x, y, th = self.s[:3]
        dx, dy = self.s[j] - x, self.s[j + 1] - y
        q = dx * dx + dy * dy
        r = math.sqrt(q)
        z_hat = np.array([r, wrap(math.atan2(dy, dx) - th)])
        H = np.zeros((2, len(self.s)))
        H[:, :3] = [[-dx / r, -dy / r, 0.0],
                    [dy / q, -dx / q, -1.0]]
        H[:, j:j + 2] = [[dx / r, dy / r],
                         [-dy / q, dx / q]]
        return z_hat, H

    # ---------------- update / augment ----------------
    def observe(self, tag_id, r, b):
        """
        One sighting of tag_id at range r (m) and bearing b (rad, + = left).
        Returns 'new', 'updated', or 'rejected' (failed the validation gate)
        along with the Mahalanobis distance (None for a new landmark).
        """
        if tag_id not in self.ids:
            self.add_landmark(tag_id, r, b)
            return 'new', None
        j = self._idx(tag_id)
        z_hat, H = self.h(j)
        nu = np.array([r - z_hat[0], wrap(b - z_hat[1])])     # innovation
        R = self.R_for(r)
        S = H @ self.Sigma @ H.T + R
        S_inv = np.linalg.inv(S)
        m = float(nu @ S_inv @ nu)
        if m > self.gate:
            return 'rejected', m
        K = self.Sigma @ H.T @ S_inv
        self.s = self.s + K @ nu
        self.s[2] = wrap(self.s[2])
        IKH = np.eye(len(self.s)) - K @ H
        self.Sigma = IKH @ self.Sigma @ IKH.T + K @ R @ K.T    # Joseph form
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)
        return 'updated', m

    def add_landmark(self, tag_id, r, b):
        """
        Augment the state with a new landmark placed where it was seen:
            z_new = p + r [cos(theta + b), sin(theta + b)]
        Its covariance comes from the robot uncertainty (through G_x) plus the
        measurement noise (through G_z), and its cross-covariance with the
        robot and every other landmark is G_x times the robot's rows.
        """
        x, y, th = self.s[:3]
        a = th + b
        ca, sa = math.cos(a), math.sin(a)
        z_new = np.array([x + r * ca, y + r * sa])
        Gx = np.array([[1.0, 0.0, -r * sa],
                       [0.0, 1.0, r * ca]])
        Gz = np.array([[ca, -r * sa],
                       [sa, r * ca]])
        n = len(self.s)
        S = self.Sigma
        S_new = np.zeros((n + 2, n + 2))
        S_new[:n, :n] = S
        cross = Gx @ S[:3, :]                                   # 2 x n
        S_new[n:, :n] = cross
        S_new[:n, n:] = cross.T
        S_new[n:, n:] = Gx @ S[:3, :3] @ Gx.T + Gz @ self.R_for(r) @ Gz.T
        self.s = np.concatenate([self.s, z_new])
        self.Sigma = S_new
        self.ids.append(tag_id)


def make_ekf(p):
    """Build the filter from a dict of the ROS parameters (also used by replay_slam.py)."""
    return EKFSlam(k_s=p['k_s'], k_theta=p['k_theta'], k_s_theta=p['k_s_theta'],
                   min_sigma_s=p['min_sigma_s'], min_sigma_theta=p['min_sigma_theta'],
                   sigma_r0=p['sigma_r0'], k_r=p['k_r'],
                   sigma_b=math.radians(p['sigma_b_deg']), gate=p['gate'])


EKF_PARAMS = ['k_s', 'k_theta', 'k_s_theta', 'min_sigma_s', 'min_sigma_theta',
              'sigma_r0', 'k_r', 'sigma_b_deg', 'gate']


def detection_to_range_bearing(fx, ly):
    """Tag center in the robot frame (forward, left) -> (range, bearing)."""
    return math.hypot(fx, ly), math.atan2(ly, fx)


def sighting_position(pose, r, b):
    """Where one sighting (r, b) puts the tag, given the robot pose (x, y, theta)."""
    x, y, th = pose
    return x + r * math.cos(th + b), y + r * math.sin(th + b)


def _corners(name, size):
    """
    Corners (x, y, heading on arrival) of one counter-clockwise loop that
    starts and ends at (0, 0, 0): drive one side, turn left, repeat.
    """
    if name == 'square':
        sides = 4
    elif name == 'octagon':
        sides = 8
    else:
        raise ValueError(f'Unknown path "{name}" (use square / octagon)')
    x = y = 0.0
    out = []
    for k in range(sides):
        h = k * 2 * math.pi / sides
        x += size * math.cos(h)
        y += size * math.sin(h)
        out.append((round(x, 6), round(y, 6), wrap((k + 1) * 2 * math.pi / sides)))
    out[-1] = (0.0, 0.0, 0.0)            # close the loop exactly
    return out


def path_parts(name, size, centered=True):
    """
    (entry, one_loop) waypoint lists for 'square' (n x n) or 'octagon' (side m).

    centered=False: the start pose is a corner of the path; the robot needs
        room AHEAD and to its LEFT (n for the square, 2.41 * m for the octagon).
    centered=True:  the path is centered on the start pose, so it only needs
        about n/2 (square) or 1.31 * m (octagon corners) of room in every
        direction. The robot first drives STRAIGHT AHEAD to the middle of the
        side in front of it (entry), turns left, and goes around from there;
        each loop ends back at that mid-side point.
    """
    corners = _corners(name, size)
    if not centered:
        return [], corners
    cx = sum(c[0] for c in corners) / len(corners)
    cy = sum(c[1] for c in corners) / len(corners)
    shifted = [(round(x - cx, 6), round(y - cy, 6), th) for x, y, th in corners]
    # the side straight ahead of the start is the one driven at heading +90 deg
    # (each corner stores the heading of the side that starts there)
    n = len(shifted)
    i = next(k for k in range(n) if abs(wrap(shifted[k][2] - math.pi / 2)) < 1e-6)
    a, b = shifted[i], shifted[(i + 1) % n]
    mid = (round((a[0] + b[0]) / 2, 6), round((a[1] + b[1]) / 2, 6), math.pi / 2)
    loop = [shifted[(i + 1 + k) % n] for k in range(n)] + [mid]
    return [mid], loop


def square_path(n, loops=1, centered=True):
    """Waypoints (x, y, theta) for an n x n square, counter-clockwise, from (0, 0, 0)."""
    entry, loop = path_parts('square', n, centered)
    return [(0.0, 0.0, 0.0)] + entry + loop * loops


def octagon_path(m, loops=1, centered=True):
    """
    Waypoints for a regular octagon with side m (drive m, turn left 45 deg,
    8 times), counter-clockwise, from (0, 0, 0).
    """
    entry, loop = path_parts('octagon', m, centered)
    return [(0.0, 0.0, 0.0)] + entry + loop * loops
