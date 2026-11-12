#!/usr/bin/env python3
"""
HW3 part 4: compare the EKF map with the measured ground-truth map.
No ROS needed (numpy + matplotlib).

1) Write the measured map as a CSV in the WORLD frame (e.g. origin at one
   corner of the area, x along one wall, y along the other):
       id,x,y
       0,3.0,6.5
       ...
2) Measure the robot's start pose in that same frame: position of the robot
   CENTER and the direction it faces (degrees, CCW from world +x).
3) Run:
       python3 analyze_map.py LOG_DIR --gt ground_truth.csv \
           --start 3.0 2.0 90 --units ft --room 10
   LOG_DIR is a hw3 run folder (or its replay/ subfolder).

The EKF map lives in the robot's start frame, so every estimate is moved
into the world frame with the measured start pose:
       p_world = R(theta0) p_map + (x0, y0)
Then the error of each landmark is its distance to the measured position.

Also reported: the error after the best rigid fit (rotation + shift)
between the two maps. If that is much smaller than the start-pose error,
most of the error came from measuring the start pose, not the map shape.

Just the ground-truth sketch for part 1 (no run needed):
       python3 analyze_map.py --gt ground_truth.csv --start 3 2 90 --units ft \
           --room 10 --sketch

Just the robot's own map (no ground truth needed), in the start-pose frame:
       python3 analyze_map.py LOG_DIR
   Writes slam_map.png: estimated tags with 2-sigma ellipses, every single
   sighting (from observations.csv) and the estimated robot path.
"""
import argparse
import csv
import math
import os

import numpy as np

UNITS = {'m': 1.0, 'ft': 0.3048, 'in': 0.0254, 'cm': 0.01}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_gt(path, scale):
    return {int(r['id']): np.array([float(r['x']), float(r['y'])]) * scale
            for r in read_csv(path)}


def load_map(path):
    out = {}
    for r in read_csv(path):
        S = np.array([[float(r['sxx']), float(r['sxy'])], [float(r['sxy']), float(r['syy'])]])
        out[int(r['id'])] = (np.array([float(r['x']), float(r['y'])]), S)
    return out


def to_world(start):
    x0, y0, th0 = start
    R = np.array([[math.cos(th0), -math.sin(th0)], [math.sin(th0), math.cos(th0)]])
    t = np.array([x0, y0])
    return R, t


def best_rigid_fit(A, B):
    """Rotation R and shift t minimizing sum |R a + t - b|^2 (2D Kabsch)."""
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def compare(est, gt, R, t):
    """Per-tag rows (id, est_world_x, est_world_y, gt_x, gt_y, err) for shared IDs."""
    rows = []
    for tid in sorted(set(est) & set(gt)):
        pw = R @ est[tid][0] + t
        rows.append((tid, pw[0], pw[1], gt[tid][0], gt[tid][1],
                     float(np.linalg.norm(pw - gt[tid]))))
    return rows


def ellipse(ax, center, S, R, color, n_sigma=2.0):
    S_w = R @ S @ R.T
    vals, vecs = np.linalg.eigh(S_w)
    a = np.linspace(0, 2 * np.pi, 60)
    pts = vecs @ (n_sigma * np.sqrt(np.maximum(vals, 0))[:, None] * np.vstack([np.cos(a), np.sin(a)]))
    ax.plot(center[0] + pts[0], center[1] + pts[1], color=color, lw=0.8, alpha=0.7)


def draw_base(ax, gt, start, room):
    if room:
        ax.plot([0, room, room, 0, 0], [0, 0, room, room, 0], 'k-', lw=1, alpha=0.5)
    g = np.array(list(gt.values()))
    ax.scatter(g[:, 0], g[:, 1], marker='s', s=60, c='tab:green', label='measured tags')
    for tid, p in gt.items():
        ax.annotate(str(tid), p, textcoords='offset points', xytext=(6, 6), fontsize=9,
                    color='tab:green')
    x0, y0, th0 = start
    ax.arrow(x0, y0, 0.25 * math.cos(th0), 0.25 * math.sin(th0), width=0.02,
             color='k', length_includes_head=True)
    ax.plot(x0, y0, 'ko', ms=4, label='start pose')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')


def plot_slam_map(log_dir, out, plt):
    """The robot's own map in its start frame: no ground truth needed."""
    est = load_map(os.path.join(log_dir, 'map_final.csv'))
    os.makedirs(out, exist_ok=True)
    print(f'EKF found {len(est)} landmarks (map frame = start pose, meters)')
    print(' tag       x        y    2-sigma x / y (cm)')
    for tid, (p, S) in sorted(est.items()):
        print(f'{tid:4d}  {p[0]:+7.3f}  {p[1]:+7.3f}     {200 * math.sqrt(max(S[0, 0], 0)):5.1f} / '
              f'{200 * math.sqrt(max(S[1, 1], 0)):5.1f}')

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    col = {tid: colors[i % len(colors)] for i, tid in enumerate(sorted(est))}
    obs_path = os.path.join(log_dir, 'observations.csv')
    if os.path.exists(obs_path):
        for r in read_csv(obs_path):
            tid = int(r['id'])
            ax.plot(float(r['seen_x']), float(r['seen_y']), '.', ms=4, alpha=0.35,
                    color=col.get(tid, 'gray'))
        ax.plot([], [], '.', color='gray', label='single sightings')
    traj_path = os.path.join(log_dir, 'trajectory.csv')
    if os.path.exists(traj_path):
        tr = np.array([[float(r['x']), float(r['y'])] for r in read_csv(traj_path)])
        ax.plot(tr[:, 0], tr[:, 1], '-', color='tab:blue', lw=1, alpha=0.7,
                label='estimated robot path')
    I = np.eye(2)
    for tid, (p, S) in est.items():
        ax.plot(p[0], p[1], 'x', color=col[tid], ms=9, mew=2)
        ellipse(ax, p, S, I, col[tid])
        ax.annotate(str(tid), p, textcoords='offset points', xytext=(6, 6), fontsize=9,
                    color=col[tid])
    ax.plot([], [], 'x', color='k', ms=8, mew=2, label='EKF tags (2-sigma)')
    ax.arrow(0, 0, 0.25, 0, width=0.02, color='k', length_includes_head=True)
    ax.plot(0, 0, 'ko', ms=4, label='start pose (0, 0, 0)')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_xlabel('x (m), start heading')
    ax.set_ylabel('y (m), left of start')
    ax.set_title(f'EKF SLAM map: {len(est)} landmarks')
    ax.legend(loc='upper right', fontsize=8)
    path = os.path.join(out, 'slam_map.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Wrote {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('log_dir', nargs='?')
    ap.add_argument('--gt', help='measured tags CSV: id,x,y (omit to plot only the EKF map)')
    ap.add_argument('--start', nargs=3, type=float,
                    metavar=('X0', 'Y0', 'THETA0_DEG'), help='measured start pose')
    ap.add_argument('--units', default='m', choices=UNITS, help='units of --gt and --start')
    ap.add_argument('--room', type=float, default=0.0, help='side of the square area')
    ap.add_argument('--sketch', action='store_true', help='only draw the ground-truth map')
    ap.add_argument('--out', default=None, help='output folder (default: LOG_DIR)')
    a = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not a.gt:
        if not a.log_dir:
            ap.error('give LOG_DIR (to plot the EKF map) and/or --gt')
        plot_slam_map(a.log_dir, a.out or a.log_dir, plt)
        return
    if a.start is None:
        ap.error('--start X0 Y0 THETA0_DEG is needed with --gt')

    k = UNITS[a.units]
    gt = load_gt(a.gt, k)
    start = (a.start[0] * k, a.start[1] * k, math.radians(a.start[2]))
    room = a.room * k
    out = a.out or a.log_dir or '.'
    os.makedirs(out, exist_ok=True)

    if a.sketch or not a.log_dir:
        fig, ax = plt.subplots(figsize=(6, 6))
        draw_base(ax, gt, start, room)
        ax.set_title('Ground-truth map (top-down)')
        ax.legend(loc='upper right', fontsize=8)
        path = os.path.join(out, 'ground_truth_map.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Wrote {path}')
        return

    est = load_map(os.path.join(a.log_dir, 'map_final.csv'))
    R, t = to_world(start)
    rows = compare(est, gt, R, t)
    if not rows:
        raise SystemExit('No tag IDs in common between the EKF map and --gt')
    errs = np.array([r[5] for r in rows])

    print(f'EKF found {len(est)} landmarks: {sorted(est)}')
    if set(est) - set(gt):
        print(f'  in the EKF map but not measured: {sorted(set(est) - set(gt))}')
    if set(gt) - set(est):
        print(f'  measured but never seen: {sorted(set(gt) - set(est))}')
    print('\n tag   est_x   est_y    gt_x    gt_y   error (cm)')
    for tid, ex, ey, gx, gy, e in rows:
        print(f'{tid:4d} {ex:7.3f} {ey:7.3f} {gx:7.3f} {gy:7.3f}   {e * 100:6.1f}')
    print(f'\nMean landmark error (measured start pose): {errs.mean() * 100:.1f} cm '
          f'(max {errs.max() * 100:.1f} cm)')

    shared = [r[0] for r in rows]
    fit_note = ''
    if len(shared) >= 2:
        Rf, tf = best_rigid_fit(np.array([est[i][0] for i in shared]),
                                np.array([gt[i] for i in shared]))
        fit_errs = np.array([r[5] for r in compare(est, gt, Rf, tf)])
        dth = math.degrees(math.atan2(Rf[1, 0], Rf[0, 0])) - a.start[2]
        fit_note = (f'Mean error after best rigid fit: {fit_errs.mean() * 100:.1f} cm '
                    f'(the fit says the start heading was off by {(dth + 180) % 360 - 180:+.1f} deg)')
        print(fit_note)

    with open(os.path.join(out, 'map_errors.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'est_world_x', 'est_world_y', 'gt_x', 'gt_y', 'error_m'])
        w.writerows(rows)
        w.writerow([])
        w.writerow(['mean_error_m', errs.mean()])

    # per-loop errors (does driving more loops help?)
    loops_path = os.path.join(a.log_dir, 'map_loops.csv')
    loop_rows = read_csv(loops_path) if os.path.exists(loops_path) else []
    loop_err = []
    if loop_rows:
        keys = []
        for r in loop_rows:
            key = (r['path'], r['loop'])
            if key not in keys:
                keys.append(key)
        print('\nAfter each loop:')
        for key in keys:
            m = {int(r['id']): (np.array([float(r['x']), float(r['y'])]), None)
                 for r in loop_rows if (r['path'], r['loop']) == key}
            e = [c[5] for c in compare(m, gt, R, t)]
            if e:
                loop_err.append((f'{key[0]} {key[1]}', float(np.mean(e)), len(m)))
                print(f'  {key[0]} loop {key[1]}: {len(m)} landmarks, mean error '
                      f'{np.mean(e) * 100:.1f} cm')

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_base(ax, gt, start, room)
    traj_path = os.path.join(a.log_dir, 'trajectory.csv')
    if os.path.exists(traj_path):
        tr = np.array([[float(r['x']), float(r['y'])] for r in read_csv(traj_path)])
        tw = tr @ R.T + t
        ax.plot(tw[:, 0], tw[:, 1], '-', color='tab:blue', lw=1, alpha=0.7,
                label='estimated robot path')
    for tid, (p, S) in est.items():
        pw = R @ p + t
        ax.plot(pw[0], pw[1], 'x', color='tab:red', ms=8, mew=2)
        ellipse(ax, pw, S, R, 'tab:red')
        if tid in gt:
            ax.plot([pw[0], gt[tid][0]], [pw[1], gt[tid][1]], ':', color='gray', lw=1)
    ax.plot([], [], 'x', color='tab:red', ms=8, mew=2, label='EKF tags (2-sigma)')
    ax.set_title(f'EKF map vs measured map: mean error {errs.mean() * 100:.1f} cm')
    ax.legend(loc='upper right', fontsize=8)
    path = os.path.join(out, 'map_compare.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'\nWrote {path} and map_errors.csv')

    if len(loop_err) > 1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(range(1, len(loop_err) + 1), [e * 100 for _, e, _ in loop_err], 'o-')
        ax.set_xticks(range(1, len(loop_err) + 1), [n for n, _, _ in loop_err])
        ax.set_ylabel('mean landmark error (cm)')
        ax.set_title('Map error after each loop')
        ax.grid(alpha=0.3)
        path = os.path.join(out, 'error_per_loop.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Wrote {path}')


if __name__ == '__main__':
    main()
