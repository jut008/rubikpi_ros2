#!/usr/bin/env python3
"""
Re-run EKF SLAM offline on a logged hw3_slam run, so you can tune the noise
settings without driving the robot again. No ROS needed.

    python3 replay_slam.py ~/hw3_logs/hw3_20261105_170000
    python3 replay_slam.py LOG_DIR --k_theta 0.15 --sigma_b_deg 3 --out replay_a

Uses the run's params.json, with any --name value overrides. Writes
map_final.csv, map_loops.csv, trajectory.csv and observations.csv (same
columns as the robot) into LOG_DIR/<out>/, so analyze_map.py works on it
the same way. Also writes a checkpoint after every loop and at the end
(LOG_DIR/<out>/checkpoints/<name>/ + checkpoints.csv), as the robot does.
"""
import argparse
import copy
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from robot_control.ekf_slam import EKF_PARAMS, make_ekf, sighting_position  # noqa: E402
from robot_control.slam_logs import (MAP_HEADER, OBS_HEADER, TRAJ_HEADER,  # noqa: E402
                                     landmark_rows, write_checkpoint, write_csv)


def replay(log_dir, overrides=None):
    with open(os.path.join(log_dir, 'params.json')) as f:
        p = json.load(f)
    p.update(overrides or {})
    ekf = make_ekf(p)
    traj, loop_maps, sightings, checkpoints = [], [], [], []
    counts = {'new': 0, 'updated': 0, 'rejected': 0}
    dist = rot = 0.0
    t = 0.0
    with open(os.path.join(log_dir, 'events.csv')) as f:
        rows = list(csv.DictReader(f))
    # tags in each look (a run of consecutive obs rows), for the new-tag rule
    look_ids, cur = [], []
    for row in rows:
        if row['kind'] == 'obs':
            cur.append(int(row['a']))
        elif row['kind'] != 'deferred':
            cur = []
        look_ids.append(cur)
    lim = math.radians(p.get('new_tag_max_bearing_deg', 0.0) or 0.0)
    known_in_view = False
    for k, row in enumerate(rows):
        t = float(row['t'])
        if row['kind'] in ('obs', 'deferred') and (k == 0 or rows[k - 1]['kind']
                                                   not in ('obs', 'deferred')):
            known_in_view = any(i in ekf.ids for i in look_ids[k])
        if row['kind'] == 'predict':
            ekf.predict(float(row['a']), float(row['b']))
            dist += abs(float(row['a']))
            rot += abs(float(row['b']))
        elif row['kind'] in ('obs', 'deferred'):
            tid, r, b = int(row['a']), float(row['b']), float(row['c'])
            if row['kind'] == 'deferred':
                continue              # the robot skipped it too
            if lim > 0 and known_in_view and tid not in ekf.ids and abs(b) > lim:
                continue              # same rule as the robot, for old logs
            pose = ekf.pose
            seen = sighting_position(pose, r, b)
            result, m = ekf.observe(tid, r, b)
            counts[result] += 1
            est = ekf.landmarks()[tid]
            sightings.append([row['t'], '', tid, r, np.degrees(b), pose[0], pose[1],
                              np.degrees(pose[2]), *seen, result,
                              '' if m is None else m, est[0], est[1]])
        elif row['kind'] == 'loop_end':
            for tid, (x, y, S) in ekf.landmarks().items():
                loop_maps.append([row['a'], row['b'], tid, x, y,
                                  S[0, 0], S[0, 1], S[1, 1]])
        x, y, th = ekf.pose
        sd = np.sqrt(np.maximum(np.diag(ekf.Sigma)[:3], 0.0))
        traj.append([row['t'], x, y, th, *sd, ekf.n_landmarks, ''])
        if row['kind'] == 'loop_end':
            # a frozen copy of the filter + how many log rows exist at this moment
            checkpoints.append({'name': f"{row['a']}_loop{row['b']}", 'path': row['a'],
                                'loop': row['b'], 't': t, 'ekf': copy.deepcopy(ekf),
                                'n_traj': len(traj), 'n_obs': len(sightings),
                                'dist': dist, 'rot': rot})
    checkpoints.append({'name': 'final', 'path': '', 'loop': '', 't': t, 'ekf': ekf,
                        'n_traj': len(traj), 'n_obs': len(sightings), 'dist': dist, 'rot': rot})
    return ekf, traj, loop_maps, sightings, counts, p, checkpoints


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('log_dir')
    ap.add_argument('--out', default='replay')
    for n in EKF_PARAMS + ['new_tag_max_bearing_deg']:
        ap.add_argument(f'--{n}', type=float)
    a = ap.parse_args()
    over = {n: getattr(a, n) for n in EKF_PARAMS + ['new_tag_max_bearing_deg']
            if getattr(a, n) is not None}
    ekf, traj, loop_maps, sightings, counts, p, checkpoints = replay(a.log_dir, over)

    out = os.path.join(a.log_dir, a.out)
    os.makedirs(out, exist_ok=True)

    def write(name, header, rows):
        write_csv(out, name, header, rows)

    write('trajectory.csv', TRAJ_HEADER, traj)
    write('map_loops.csv', ['path', 'loop', 'id', 'x', 'y', 'sxx', 'sxy', 'syy'], loop_maps)
    write('map_final.csv', MAP_HEADER, landmark_rows(ekf))
    write('observations.csv', OBS_HEADER, sightings)
    index = os.path.join(out, 'checkpoints.csv')
    if os.path.exists(index):
        os.remove(index)                  # a fresh index for this replay
    for c in checkpoints:
        write_checkpoint(out, c['name'], c['ekf'], traj[:c['n_traj']], sightings[:c['n_obs']],
                         c['t'], c['path'], c['loop'], c['dist'], c['rot'])
    with open(os.path.join(out, 'params.json'), 'w') as f:
        json.dump(p, f, indent=2)
    settings = {n: p.get(n) for n in EKF_PARAMS}
    print(f'EKF settings: {settings}')
    print(f'Sightings: {counts}')
    print(f'{ekf.n_landmarks} landmarks; final pose '
          f'({ekf.pose[0]:+.3f}, {ekf.pose[1]:+.3f}, {np.degrees(ekf.pose[2]):+.1f} deg)')
    for tid, (x, y, S) in sorted(ekf.landmarks().items()):
        print(f'  tag {tid:3d}: ({x:+.3f}, {y:+.3f})  2-sigma '
              f'{2 * np.sqrt(S[0, 0]) * 100:.1f} x {2 * np.sqrt(S[1, 1]) * 100:.1f} cm')
    print(f'Checkpoints: {", ".join(c["name"] for c in checkpoints)}')
    print(f'Wrote {out}/')


if __name__ == '__main__':
    main()
