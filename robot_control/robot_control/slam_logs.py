"""
Log writing shared by hw3_slam.py (on the robot) and replay_slam.py (offline).
Plain Python + numpy, no ROS.

Checkpoints: a snapshot of the whole estimate at one moment, written as

    LOG_DIR/checkpoints/<name>/
        map_final.csv     landmarks at that moment (same columns as the run's map_final.csv)
        trajectory.csv    estimated robot path up to that moment
        observations.csv  every tag sighting up to that moment
        state.json        the full EKF state vector s, covariance Sigma, landmark IDs
        summary.txt       readable landmark table + robot pose
    LOG_DIR/checkpoints.csv   one row per checkpoint (robot pose, uncertainty, totals)

The robot writes one after every loop (e.g. square_loop1, square_loop2) and a
'final' one at the end. Each checkpoint folder has the same files as a run
folder, so analyze_map.py works on it directly:

    python3 analyze_map.py LOG_DIR/checkpoints/square_loop1
"""
import csv
import json
import math
import os

import numpy as np

TRAJ_HEADER = ['t', 'x', 'y', 'theta', 'sd_x', 'sd_y', 'sd_theta', 'n_landmarks', 'waypoint']
OBS_HEADER = ['t', 'waypoint', 'id', 'range_m', 'bearing_deg', 'robot_x', 'robot_y',
              'robot_theta_deg', 'seen_x', 'seen_y', 'result', 'mahalanobis', 'est_x', 'est_y']
MAP_HEADER = ['id', 'x', 'y', 'sxx', 'sxy', 'syy']
INDEX_HEADER = ['checkpoint', 'path', 'loop', 't', 'robot_x', 'robot_y', 'robot_theta_deg',
                'sd_x', 'sd_y', 'sd_theta_deg', 'n_landmarks', 'distance_m', 'rotation_deg']
OBS_RESULT_COL = OBS_HEADER.index('result')


def write_csv(folder, name, header, rows):
    with open(os.path.join(folder, name), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def landmark_rows(ekf):
    return [[tid, x, y, S[0, 0], S[0, 1], S[1, 1]]
            for tid, (x, y, S) in ekf.landmarks().items()]


def pose_and_sd(ekf):
    x, y, th = ekf.pose
    sd = np.sqrt(np.maximum(np.diag(ekf.Sigma)[:3], 0.0))
    return (x, y, th), sd


def map_table(ekf, sightings):
    """Readable lines: every landmark's position, uncertainty and sighting counts."""
    counts = {}
    for row in sightings:
        c = counts.setdefault(int(row[2]), {'new': 0, 'updated': 0, 'rejected': 0})
        c[row[OBS_RESULT_COL]] += 1
    lines = [' tag      x (m)    y (m)   2-sigma x/y (cm)   seen  used  rejected']
    for tid, (x, y, S) in sorted(ekf.landmarks().items()):
        c = counts.get(tid, {'new': 0, 'updated': 0, 'rejected': 0})
        lines.append(f'{tid:4d}   {x:+8.3f} {y:+8.3f}   '
                     f'{200 * math.sqrt(max(S[0, 0], 0)):6.1f} / '
                     f'{200 * math.sqrt(max(S[1, 1], 0)):5.1f}     '
                     f'{sum(c.values()):4d}  {c["new"] + c["updated"]:4d}  {c["rejected"]:6d}')
    return lines


def write_checkpoint(root, name, ekf, traj, sightings, t, path='', loop='',
                     distance=0.0, rotation=0.0, title=None):
    """
    Write checkpoints/<name>/ under root and add a row to root/checkpoints.csv.
    traj and sightings are the rows logged so far. Returns the folder path.
    """
    folder = os.path.join(root, 'checkpoints', name)
    os.makedirs(folder, exist_ok=True)
    write_csv(folder, 'map_final.csv', MAP_HEADER, landmark_rows(ekf))
    write_csv(folder, 'trajectory.csv', TRAJ_HEADER, traj)
    write_csv(folder, 'observations.csv', OBS_HEADER, sightings)
    with open(os.path.join(folder, 'state.json'), 'w') as f:
        json.dump({'checkpoint': name, 't': t, 'landmark_ids': list(ekf.ids),
                   's': ekf.s.tolist(), 'Sigma': ekf.Sigma.tolist()}, f)

    (x, y, th), sd = pose_and_sd(ekf)
    summary = [
        title or f'Checkpoint {name}',
        f'time {t:.1f} s, driven {distance:.2f} m, turned {math.degrees(rotation):.0f} deg',
        '',
        'Map frame = robot start pose: origin at the start position, +x = starting',
        'heading, +y = 90 deg to its left. Positions in meters.',
        '',
        f'Landmarks found: {ekf.n_landmarks}',
        *map_table(ekf, sightings),
        '',
        f'Robot pose estimate: ({x:+.3f}, {y:+.3f}, {math.degrees(th):+.1f} deg)  '
        f'1-sigma ({sd[0] * 100:.1f} cm, {sd[1] * 100:.1f} cm, {math.degrees(sd[2]):.1f} deg)',
    ]
    with open(os.path.join(folder, 'summary.txt'), 'w') as f:
        f.write('\n'.join(summary) + '\n')

    index = os.path.join(root, 'checkpoints.csv')
    new = not os.path.exists(index)
    with open(index, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(INDEX_HEADER)
        w.writerow([name, path, loop, round(t, 2), x, y, math.degrees(th), sd[0], sd[1],
                    math.degrees(sd[2]), ekf.n_landmarks, distance, math.degrees(rotation)])
    return folder
