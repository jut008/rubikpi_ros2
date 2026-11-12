"""
Simulated check of ekf_slam.py (no robot needed):
    python3 -m pytest robot_control/test/test_ekf_slam.py -q
A robot with biased, noisy odometry drives the square and octagon paths
(centered on its start pose) in an 8 ft x 8 ft area with two tags on each
side, from the center and from an off-center, rotated start. The EKF map
should beat dead reckoning and land within a few cm of the true tags.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from robot_control.ekf_slam import EKFSlam, octagon_path, square_path, wrap  # noqa: E402

ROOM = 8 * 0.3048            # 8 ft square, world frame origin at one corner
A, B = ROOM / 3, 2 * ROOM / 3
TAGS_WORLD = {0: (A, 0.0), 1: (B, 0.0), 2: (ROOM, A), 3: (ROOM, B),
              4: (B, ROOM), 5: (A, ROOM), 6: (0.0, B), 7: (0.0, A)}
STARTS = {'center': (ROOM / 2, ROOM / 2, 0.0),
          'off-center': (ROOM / 2 - 0.2, ROOM / 2 + 0.15, math.radians(120))}
FOV = math.radians(35)       # half field of view
MAX_RANGE = 3.0


def tags_in_start_frame(start):
    """The robot doesn't know its start pose: its map frame IS the start pose."""
    x0, y0, th0 = start
    c, s = math.cos(th0), math.sin(th0)
    return {t: (c * (x - x0) + s * (y - y0), -s * (x - x0) + c * (y - y0))
            for t, (x, y) in TAGS_WORLD.items()}


def simulate(waypoints, start, seed=0, step=0.25):
    TAGS = tags_in_start_frame(start)
    rng = np.random.default_rng(seed)
    true = np.array([0.0, 0.0, 0.0])
    ekf = EKFSlam()
    odo = np.array([0.0, 0.0, 0.0])

    def sense():
        for tid, (tx, ty) in TAGS.items():
            dx, dy = tx - true[0], ty - true[1]
            r = math.hypot(dx, dy)
            b = wrap(math.atan2(dy, dx) - true[2])
            if r < MAX_RANGE and abs(b) < FOV:
                ekf.observe(tid, r + rng.normal(0, 0.01 + 0.02 * r),
                            b + rng.normal(0, math.radians(1.5)))

    def move(ds, dth):
        # the real robot under-turns by 5% and drives 4% long, plus noise
        real_ds = ds * 1.04 + rng.normal(0, 0.02 * abs(ds))
        real_dth = dth * 0.95 + rng.normal(0, 0.03 * abs(dth) + 0.01 * abs(ds))
        thm = true[2] + real_dth / 2
        true[0] += real_ds * math.cos(thm)
        true[1] += real_ds * math.sin(thm)
        true[2] = wrap(true[2] + real_dth)
        ekf.predict(ds, dth)
        thm = odo[2] + dth / 2
        odo[0] += ds * math.cos(thm)
        odo[1] += ds * math.sin(thm)
        odo[2] = wrap(odo[2] + dth)
        sense()

    sense()
    for wp in waypoints[1:]:
        for _ in range(20):                    # a few correction steps per waypoint
            x, y, th = ekf.pose
            dx, dy = wp[0] - x, wp[1] - y
            d = math.hypot(dx, dy)
            if d < 0.05:
                break
            err = wrap(math.atan2(dy, dx) - th)
            if abs(err) > 0.1:
                move(0.0, err)
            else:
                move(min(step, d), 0.0)
        move(0.0, wrap(wp[2] - ekf.pose[2]))
    lm = ekf.landmarks()
    err = np.mean([math.hypot(lm[t][0] - TAGS[t][0], lm[t][1] - TAGS[t][1]) for t in lm])
    pose_err = math.hypot(ekf.pose[0] - true[0], ekf.pose[1] - true[1])
    odo_err = math.hypot(odo[0] - true[0], odo[1] - true[1])
    return ekf, err, pose_err, odo_err


def test_centered_paths_stay_in_the_room():
    for start in STARTS.values():
        x0, y0, th0 = start
        for wps in [square_path(1.0, loops=1), octagon_path(0.6, loops=1)]:
            for x, y, _ in wps:
                wx = x0 + x * math.cos(th0) - y * math.sin(th0)
                wy = y0 + x * math.sin(th0) + y * math.cos(th0)
                # robot center stays 0.2 m from the walls (robot half-length + margin)
                assert 0.2 < wx < ROOM - 0.2 and 0.2 < wy < ROOM - 0.2


def test_square_and_octagon():
    for where, start in STARTS.items():
        for name, wps in [('square', square_path(1.0, loops=2)),
                          ('octagon', octagon_path(0.6, loops=2))]:
            ekf, err, pose_err, odo_err = simulate(wps, start)
            print(f'{where} {name}: {ekf.n_landmarks} tags, mean map error '
                  f'{err * 100:.1f} cm, final pose error {pose_err * 100:.1f} cm '
                  f'(odometry only {odo_err * 100:.1f} cm)')
            check(ekf, err, pose_err, odo_err)


def check(ekf, err, pose_err, odo_err):
    assert ekf.n_landmarks >= 6
    assert err < 0.15
    assert pose_err < odo_err
    assert np.all(np.linalg.eigvalsh(ekf.Sigma) > -1e-9)   # still a covariance


if __name__ == '__main__':
    test_centered_paths_stay_in_the_room()
    test_square_and_octagon()
