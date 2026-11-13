#!/usr/bin/env python3
"""
Calibrate the HW3 robot's timed motions against the taped tag map:
forward steps, 45 deg turns and 90 deg turns.

The motions are exactly the ones hw3_slam.py sends: the same motor commands,
durations and settle time, all read from hw3_params.yaml. After each motion
the script takes a few camera frames and works out where the robot really
is, from the tags it sees and their taped positions in ground_truth.csv. The
difference between two poses is what the motion really did.

    forward   N drive steps of step_time each  -> drive_speed, drive_curve_deg_per_m
    turn45    45 deg pulses, left then right   -> turn_rate_left / turn_rate_right
    turn90    90 deg pulses, left then right   -> turn_rate_left / turn_rate_right
              (with both, also a "dead time": how much shorter every pulse
               comes out, which is why small turns behave differently)

Setup
    - Tags on the walls exactly where ground_truth.csv says.
    - Robot near the middle of the room. For 'forward', facing a wall with
      tags, with at least 1 m of free floor ahead.
    - At least 2 tags in view at the start of each test (1 is enough later
      for turns, but 2 is better).

Run (with ROS sourced, in other terminals:
    ros2 launch robot_vision_camera robot_vision_camera.launch.py
    ros2 run robot_control motor_control)
    python3 calibrate_hw3.py                     # all three tests
    python3 calibrate_hw3.py --test turn45 --pulses 6
    python3 calibrate_hw3.py --test forward --steps 6
Options: --params, --gt (ground truth CSV), --units (of the CSV, default ft),
--frames (camera frames averaged per look), --out (CSV of every measurement).
"""
import argparse
import csv
import math
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32MultiArray

try:
    from robot_control.tag_localization import TagDetector, tag_in_robot_frame
except ImportError:                      # run from the source folder, package not built
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tag_localization import TagDetector, tag_in_robot_frame

SRC = os.path.expanduser('~/ros2_ws/src/rubikpi_ros2')
UNITS = {'m': 1.0, 'ft': 0.3048, 'in': 0.0254, 'cm': 0.01}
SIGMA_R, SIGMA_B = 0.03, math.radians(2.0)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


# ---------------------------------------------------------------- pose from tags
def solve_pose(seen, tags, guess=None):
    """
    Robot pose (x, y, theta) in the tag-map frame from sightings
    seen = {id: (range, bearing)} of tags with known positions.
    2+ tags: full pose. 1 tag: heading only, keeping guess's position.
    Returns (pose, rms of the normalized residuals) or (None, None).
    """
    ids = [i for i in seen if i in tags]
    if not ids:
        return None, None
    if len(ids) == 1:
        if guess is None:
            return None, None
        i = ids[0]
        r, b = seen[i]
        dx, dy = tags[i] - guess[:2]
        return np.array([guess[0], guess[1], wrap(math.atan2(dy, dx) - b)]), 0.0
    if guess is None:
        # closed form from the two farthest-apart tags: the vector between them
        # is known in both the map and the robot frame -> heading, then position
        i, j = max(((a, c) for a in ids for c in ids if a != c),
                   key=lambda p: np.linalg.norm(tags[p[0]] - tags[p[1]]))
        pi = seen[i][0] * np.array([math.cos(seen[i][1]), math.sin(seen[i][1])])
        pj = seen[j][0] * np.array([math.cos(seen[j][1]), math.sin(seen[j][1])])
        vw, vr = tags[j] - tags[i], pj - pi
        th = math.atan2(vw[1], vw[0]) - math.atan2(vr[1], vr[0])
        c, s = math.cos(th), math.sin(th)
        guess = np.array([*(tags[i] - np.array([[c, -s], [s, c]]) @ pi), th])
    p = np.array(guess, dtype=float)

    def res(q):
        out = []
        for i in ids:
            r, b = seen[i]
            dx, dy = tags[i][0] - q[0], tags[i][1] - q[1]
            out += [(math.hypot(dx, dy) - r) / SIGMA_R,
                    wrap(math.atan2(dy, dx) - q[2] - b) / SIGMA_B]
        return np.array(out)

    for _ in range(20):                  # Gauss-Newton, numeric Jacobian
        f = res(p)
        J = np.zeros((len(f), 3))
        for k in range(3):
            dp = np.zeros(3)
            dp[k] = 1e-6
            J[:, k] = (res(p + dp) - f) / 1e-6
        step = np.linalg.lstsq(J, -f, rcond=None)[0]
        p += step
        if np.abs(step).max() < 1e-7:
            break
    p[2] = wrap(p[2])
    return p, float(np.sqrt(np.mean(res(p) ** 2)))


# ---------------------------------------------------------------- ROS node
class Calib(Node):
    def __init__(self, p):
        super().__init__('calibrate_hw3')
        self.p = p
        self.pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(CompressedImage, p['image_topic'], self._img, qos)
        self.create_subscription(CameraInfo, p['camera_info_topic'], self._info, 10)
        self.frame = None
        self.K = self.D = None
        self.det = TagDetector(p['tag_size'], p['tag_family'])

    def _img(self, m):
        self.frame = (time.time(), bytes(m.data))

    def _info(self, m):
        if self.K is None:
            self.K = np.array(m.k, np.float64).reshape(3, 3)
            d = np.array(m.d, np.float64)
            self.D = np.zeros(5) if (self.p['image_is_rectified'] or d.size == 0) else d

    # same motor code as hw3_slam.py
    def publish(self, left, right):
        m = Float32MultiArray()
        m.data = [float(left), float(right)]
        self.pub.publish(m)

    def stop(self):
        for _ in range(3):
            self.publish(0.0, 0.0)
            time.sleep(0.05)

    def run_for(self, left, right, duration):
        t_end = time.time() + duration
        while time.time() < t_end and rclpy.ok():
            self.publish(left, right)
            time.sleep(0.02)
        self.stop()

    def turn(self, angle):
        """Timed turn exactly like hw3_slam.turn() without the IMU. Returns seconds."""
        cmd = self.p['turn_cmd']
        left, right = (-cmd, cmd) if angle > 0 else (cmd, -cmd)
        rate = self.p['turn_rate_left' if angle > 0 else 'turn_rate_right'] or self.p['turn_rate']
        dur = abs(angle) / rate
        self.run_for(left, right, dur)
        time.sleep(self.p['settle_time'])
        return dur

    def drive(self):
        dur = max(self.p['step_time'], self.p.get('min_step_time', 0.1))
        self.run_for(self.p['drive_left'], self.p['drive_right'], dur)
        time.sleep(self.p['settle_time'])
        return dur

    def look(self, n_frames):
        """Average range/bearing per tag over n fresh frames: {id: (r, b)}."""
        got, after, samples = 0, time.time(), {}
        t_end = time.time() + 3.0 + 0.5 * n_frames
        while got < n_frames and time.time() < t_end and rclpy.ok():
            f = self.frame
            if f is None or f[0] <= after or self.K is None:
                time.sleep(0.02)
                continue
            after = f[0]
            img = cv2.imdecode(np.frombuffer(f[1], np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            got += 1
            for d in self.det.detect(img, self.K, self.D):
                if d['dist'] > self.p['max_tag_dist'] or \
                        d['candidates'][0][2] > self.p['max_reproj_err']:
                    continue
                fx, ly, _ = tag_in_robot_frame(d['candidates'][0][0], d['candidates'][0][1],
                                               (self.p['camera_offset_x'],
                                                self.p['camera_offset_y']))
                samples.setdefault(d['id'], []).append((math.hypot(fx, ly),
                                                        math.atan2(ly, fx)))
        out = {}
        for i, v in samples.items():
            if len(v) * 2 >= got:        # seen in at least half the frames
                a = np.array(v)
                out[i] = (float(a[:, 0].mean()),
                          math.atan2(np.sin(a[:, 1]).mean(), np.cos(a[:, 1]).mean()))
        return out


# ---------------------------------------------------------------- tests
class Runner:
    def __init__(self, node, tags, args, writer):
        self.n, self.tags, self.a, self.w = node, tags, args, writer
        self.pose = None

    def measure(self, label):
        seen = self.n.look(self.a.frames)
        known = {i: v for i, v in seen.items() if i in self.tags}
        pose, rms = solve_pose(known, self.tags, self.pose)
        txt = ', '.join(f'{i}({math.degrees(b):+.0f})' for i, (r, b) in sorted(known.items()))
        if pose is None:
            if self.pose is None:
                print(f'    {label}: only {len(known)} known tag(s) in view ({txt or "none"})')
            else:
                print(f'    {label}: no known tags in view; this motion is merged '
                      'with the next one')
            return None
        flag = '  (only 1 tag: heading only)' if len(known) == 1 else ''
        if rms and rms > 3:
            flag += f'  (tags disagree, rms {rms:.1f}: treat with care)'
        print(f'    {label}: pose ({pose[0]:.2f}, {pose[1]:.2f}) m, '
              f'{math.degrees(pose[2]):+.1f} deg  tags {txt}{flag}')
        self.pose = pose
        return pose

    def start(self, name):
        input(f'\n=== {name} ===  Place the robot (see notes above), then press Enter ')
        while True:
            self.pose = None              # no guess: needs 2+ tags for a full pose
            p0 = self.measure('start')
            if p0 is not None:
                return p0
            input('    Need 2+ known tags in view. Move/rotate the robot, then press Enter ')

    def forward(self):
        p = self.start(f'forward: {self.a.steps} steps of {self.n.p["step_time"]} s')
        segs, cmd_t = [], 0.0
        for k in range(1, self.a.steps + 1):
            cmd_t += self.n.drive()
            q = self.measure(f'step {k}')
            if q is None:
                continue
            c, s = math.cos(p[2]), math.sin(p[2])
            d = q[:2] - p[:2]
            fwd, side = c * d[0] + s * d[1], -s * d[0] + c * d[1]
            dth = wrap(q[2] - p[2])
            segs.append((cmd_t, fwd, side, dth))
            self.w.writerow(['forward', k, cmd_t, self.n.p['drive_speed'] * cmd_t,
                             round(fwd, 4), round(side, 4), round(math.degrees(dth), 2)])
            print(f'      moved {fwd * 100:+.1f} cm forward, {side * 100:+.1f} cm left, '
                  f'heading {math.degrees(dth):+.1f} deg  (commanded '
                  f'{self.n.p["drive_speed"] * cmd_t * 100:.1f} cm)')
            p, cmd_t = q, 0.0
        if not segs:
            print('  No measurements.')
            return
        T = sum(s[0] for s in segs)
        D = sum(s[1] for s in segs)
        TH = sum(s[3] for s in segs)
        speed = D / T
        print(f'\n  Forward: {D:.3f} m in {T:.2f} s of driving '
              f'({D / (self.n.p["drive_speed"] * T) * 100:.0f}% of commanded)')
        print(f'    drive_speed: {speed:.3f}            (now {self.n.p["drive_speed"]})')
        if D > 0.2:
            print(f'    drive_curve_deg_per_m: {math.degrees(TH) / D:.1f}  '
                  f'(now {self.n.p["drive_curve_deg_per_m"]}; + = curves left)')
        per = [s[1] / s[0] for s in segs]
        if len(per) > 1:
            print(f'    step-to-step spread: {np.std(per) / np.mean(per) * 100:.0f}% '
                  f'(1 std of speed per step)')
        return speed

    def turns(self, deg):
        p = self.start(f'turn {deg}: {self.a.pulses} left pulses, then {self.a.pulses} right')
        res = {}
        for sign, side in ((1, 'left'), (-1, 'right')):
            cmd_sum, n_pulses, rows = 0.0, 0, []
            for k in range(1, self.a.pulses + 1):
                ang = sign * math.radians(deg)
                dur = self.n.turn(ang)
                cmd_sum += ang
                n_pulses += 1
                q = self.measure(f'{side} {k}')
                if q is None:
                    continue
                # measured change, unwrapped near the commanded total
                actual = cmd_sum + wrap(wrap(q[2] - p[2]) - cmd_sum)
                rows.append((n_pulses, dur * n_pulses, cmd_sum, actual))
                self.w.writerow([f'turn{deg}_{side}', k, round(dur * n_pulses, 4),
                                 round(math.degrees(cmd_sum), 2),
                                 '', '', round(math.degrees(actual), 2)])
                print(f'      turned {math.degrees(actual):+.1f} deg for '
                      f'{math.degrees(cmd_sum):+.1f} commanded '
                      f'({actual / cmd_sum * 100:.0f}%)')
                p, cmd_sum, n_pulses = q, 0.0, 0
            if rows:
                ratio = sum(abs(r[3]) for r in rows) / sum(abs(r[2]) for r in rows)
                per = [abs(r[3] / r[2]) for r in rows]
                key = f'turn_rate_{side}'
                print(f'  {side}: {ratio * 100:.0f}% of commanded on average '
                      f'(pulses from {min(per) * 100:.0f}% to {max(per) * 100:.0f}%)')
                print(f'    {key} for {deg} deg pulses: {self.n.p[key] * ratio:.3f}  '
                      f'(now {self.n.p[key]})')
                # seconds of motor time and radians actually turned, per pulse
                res[side] = (sum(r[1] for r in rows) / sum(r[0] for r in rows),
                             sum(abs(r[3]) for r in rows) / sum(r[0] for r in rows))
        return res


def dead_time_fit(r45, r90, p):
    """Two pulse lengths -> turned = rate * (t - t0), per side."""
    print('\n=== 45 vs 90 deg ===')
    for side in ('left', 'right'):
        if side not in r45 or side not in r90:
            continue
        (t1, a1), (t2, a2) = r45[side], r90[side]
        rate = (a2 - a1) / (t2 - t1)
        t0 = t1 - a1 / rate
        key = f'turn_rate_{side}'
        print(f'  {side}: real turning speed {rate:.3f} rad/s, dead time {t0 * 1000:+.0f} ms per '
              f'pulse')
        print(f'    -> a {math.degrees(rate * t0):+.1f} deg shortfall on EVERY turn, '
              f'whatever its size' if t0 > 0 else
              f'    -> every turn overshoots by {math.degrees(-rate * t0):.1f} deg')
        print(f'    Without a dead-time option in hw3_slam, use the rate for the pulse size '
              f'you use most (45 deg if max_turn_step_deg is 45): '
              f'{key} = {p[key] * a1 / (math.radians(45)):.3f} for 45, '
              f'{p[key] * a2 / math.radians(90):.3f} for 90')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--test', default='all', choices=['all', 'forward', 'turn45', 'turn90'])
    ap.add_argument('--params', default=f'{SRC}/robot_control/config/hw3_params.yaml')
    ap.add_argument('--gt', default=f'{SRC}/hw3_logs/ground_truth.csv')
    ap.add_argument('--units', default='ft', choices=list(UNITS))
    ap.add_argument('--steps', type=int, default=5, help='forward steps')
    ap.add_argument('--pulses', type=int, default=4, help='turn pulses per direction')
    ap.add_argument('--frames', type=int, default=5, help='frames averaged per look')
    ap.add_argument('--out', default=os.path.expanduser(
        time.strftime('~/hw3_logs/calibration_%Y%m%d_%H%M%S.csv')))
    a = ap.parse_args()

    with open(a.params) as f:
        doc = yaml.safe_load(f)
    p = {'image_topic': '/camera/image_raw/compressed',
         'camera_info_topic': '/camera/camera_info', 'image_is_rectified': False,
         'tag_family': '36h11', 'max_tag_dist': 4.0, 'max_reproj_err': 5.0,
         'camera_offset_y': 0.0, 'turn_rate': 2.0, 'drive_curve_deg_per_m': 0.0}
    p.update(doc.get('hw3_slam', doc).get('ros__parameters', {}))
    tags = {}
    with open(a.gt) as f:
        for row in csv.DictReader(f):
            tags[int(row['id'])] = np.array([float(row['x']), float(row['y'])]) * UNITS[a.units]
    print(f'Params: {a.params}\n  tag_size {p["tag_size"]}, drive {p["drive_left"]}/'
          f'{p["drive_right"]} for {p["step_time"]} s steps (drive_speed {p["drive_speed"]}), '
          f'turn_cmd {p["turn_cmd"]}, turn_rate_left {p["turn_rate_left"]}, '
          f'turn_rate_right {p["turn_rate_right"]}')
    print(f'Tag map: {a.gt} ({len(tags)} tags, {a.units})')

    rclpy.init()
    node = Calib(p)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    try:
        t_end = time.time() + 10
        while (node.K is None or node.frame is None) and time.time() < t_end:
            time.sleep(0.1)
        if node.K is None:
            print('No camera_info / images. Is the camera running?')
            return
        with open(a.out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['test', 'n', 'motor_s', 'commanded', 'forward_m', 'left_m',
                        'actual_deg_or_heading_change'])
            run = Runner(node, tags, a, w)
            r45 = r90 = None
            if a.test in ('all', 'forward'):
                run.forward()
            if a.test in ('all', 'turn45'):
                r45 = run.turns(45)
            if a.test in ('all', 'turn90'):
                r90 = run.turns(90)
            if r45 and r90:
                dead_time_fit(r45, r90, p)
        print(f'\nEvery measurement: {a.out}')
        print('Calibrate on a charged battery: timed turns slow down as it drains.')
    except (KeyboardInterrupt, EOFError):
        print('\nStopped.')
    finally:
        node.stop()
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()