#!/usr/bin/env python3
"""
Re-run EKF SLAM offline on a logged hw3_slam run, so you can tune the noise
settings without driving the robot again. No ROS needed.

    python3 replay_slam.py ~/hw3_logs/hw3_20261105_170000
    python3 replay_slam.py LOG_DIR --k_theta 0.15 --sigma_b_deg 3 --out replay_a

Uses the run's params.json, with any --name value overrides. Writes
map_final.csv, map_loops.csv and trajectory.csv (same columns as the robot)
into LOG_DIR/<out>/, so analyze_map.py works on it the same way.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from robot_control.ekf_slam import EKF_PARAMS, make_ekf  # noqa: E402


def replay(log_dir, overrides=None):
    with open(os.path.join(log_dir, 'params.json')) as f:
        p = json.load(f)
    p.update(overrides or {})
    ekf = make_ekf(p)
    traj, loop_maps = [], []
    counts = {'new': 0, 'updated': 0, 'rejected': 0}
    with open(os.path.join(log_dir, 'events.csv')) as f:
        for row in csv.DictReader(f):
            if row['kind'] == 'predict':
                ekf.predict(float(row['a']), float(row['b']))
            elif row['kind'] == 'obs':
                result, _ = ekf.observe(int(row['a']), float(row['b']), float(row['c']))
                counts[result] += 1
            elif row['kind'] == 'loop_end':
                for tid, (x, y, S) in ekf.landmarks().items():
                    loop_maps.append([row['a'], row['b'], tid, x, y,
                                      S[0, 0], S[0, 1], S[1, 1]])
            x, y, th = ekf.pose
            sd = np.sqrt(np.maximum(np.diag(ekf.Sigma)[:3], 0.0))
            traj.append([row['t'], x, y, th, *sd, ekf.n_landmarks, ''])
    return ekf, traj, loop_maps, counts, p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('log_dir')
    ap.add_argument('--out', default='replay')
    for n in EKF_PARAMS:
        ap.add_argument(f'--{n}', type=float)
    a = ap.parse_args()
    over = {n: getattr(a, n) for n in EKF_PARAMS if getattr(a, n) is not None}
    ekf, traj, loop_maps, counts, p = replay(a.log_dir, over)

    out = os.path.join(a.log_dir, a.out)
    os.makedirs(out, exist_ok=True)

    def write(name, header, rows):
        with open(os.path.join(out, name), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    write('trajectory.csv', ['t', 'x', 'y', 'theta', 'sd_x', 'sd_y', 'sd_theta',
                             'n_landmarks', 'waypoint'], traj)
    write('map_loops.csv', ['path', 'loop', 'id', 'x', 'y', 'sxx', 'sxy', 'syy'], loop_maps)
    write('map_final.csv', ['id', 'x', 'y', 'sxx', 'sxy', 'syy'],
          [[tid, x, y, S[0, 0], S[0, 1], S[1, 1]]
           for tid, (x, y, S) in ekf.landmarks().items()])
    with open(os.path.join(out, 'params.json'), 'w') as f:
        json.dump(p, f, indent=2)
    settings = {n: p[n] for n in EKF_PARAMS}
    print(f'EKF settings: {settings}')
    print(f'Sightings: {counts}')
    print(f'{ekf.n_landmarks} landmarks; final pose '
          f'({ekf.pose[0]:+.3f}, {ekf.pose[1]:+.3f}, {np.degrees(ekf.pose[2]):+.1f} deg)')
    for tid, (x, y, S) in sorted(ekf.landmarks().items()):
        print(f'  tag {tid:3d}: ({x:+.3f}, {y:+.3f})  2-sigma '
              f'{2 * np.sqrt(S[0, 0]) * 100:.1f} x {2 * np.sqrt(S[1, 1]) * 100:.1f} cm')
    print(f'Wrote {out}/')


if __name__ == '__main__':
    main()
