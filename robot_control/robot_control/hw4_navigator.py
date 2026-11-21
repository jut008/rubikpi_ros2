#!/usr/bin/env python3
"""
CSE 276A HW4: drive the path from hw4_planning.py (safe = Voronoi, fast =
visibility graph) to the goal, localizing against the KNOWN tag map.

    1. Plan:      hw4_planning.plan(world, planner) -> waypoints (x, y, theta)
                  in the arena frame (origin = bottom-left corner).
    2. Localize:  the HW3 EKF, started at the measured start pose and loaded
                  with every tag at its tape-measured position (small, fixed
                  uncertainty). There is nothing to map, so it acts as EKF
                  localization: each sighting of a known tag corrects the pose.
                  Tags that aren't in the map are ignored.
    3. Drive:     the HW3 stop-and-go loop. One motion per step (turn toward the
                  waypoint, drive one step, or back up), then predict from the
                  command, take a picture, update. A waypoint stays reached once
                  reached; at a waypoint the robot turns to the heading of the
                  next leg.
    4. Safety:    before every drive the robot checks the estimated motion
                  against the obstacle map. If the step, or the straight line to
                  the waypoint, would come too close to an obstacle (e.g. after
                  a correction moved the estimate), it re-plans from where it is
                  with the same planner.

Needs (in other terminals):
    ros2 launch robot_vision_camera robot_vision_camera.launch.py
    ros2 run robot_control motor_control
Then:
    ros2 run robot_control hw4_navigator --ros-args \\
        --params-file ~/ros2_ws/src/rubikpi_ros2/robot_control/config/hw4_params.yaml \\
        -p planner:=fast          # or safe (default: whatever the YAML says)

Logs go to log_dir/hw4_<planner>_<time>/: plan.csv, trajectory.csv (x, y
columns for plot_hw4.py --traj-safe/--traj-fast), observations.csv,
events.csv, waypoints.csv, summary.txt, params.json.
"""
import json
import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32MultiArray

from robot_control.ekf_slam import detection_to_range_bearing, make_ekf, wrap
from robot_control.hw4_planning import World, path_length, plan
from robot_control.slam_logs import write_csv
from robot_control.tag_localization import TagDetector, tag_in_robot_frame

# Used when the YAML doesn't set them. The world, tags and planner settings
# have no defaults: they must come from hw4_params.yaml.
DEFAULTS = {
    'planner': 'safe',
    'image_topic': '/camera/image_raw/compressed',
    'camera_info_topic': '/camera/camera_info',
    'image_is_rectified': False,
    'tag_family': '36h11',
    'camera_offset_x': 0.10, 'camera_offset_y': 0.0,
    # detections (separate from max_tag_dist, which is the planner's camera model)
    'detect_max_dist': 4.0,
    'max_reproj_err': 5.0,
    # motion (HW3 calibration)
    'drive_left': 0.32, 'drive_right': 0.40, 'drive_speed': 0.40,
    'drive_curve_deg_per_m': 15.0,
    'turn_cmd': 0.3, 'turn_rate': 2.0, 'turn_rate_left': 1.81, 'turn_rate_right': 1.88,
    'step_time': 0.25, 'min_step_time': 0.10, 'settle_time': 0.4,
    # waypoint following (HW3)
    'pos_tol': 0.05, 'reach_radius': 0.10, 'ang_tol': 0.20, 'heading_tol': 0.25,
    'max_turn_step_deg': 45.0, 'back_up_deg': 90.0, 'max_steps_per_waypoint': 60,
    'goal_tol': 0.08,                 # m, how close the last waypoint must be
    # safety / re-planning
    'replan': True, 'max_replans': 5,
    # EKF (HW3 values) + known-map uncertainty
    'k_s': 0.05, 'k_theta': 0.25, 'k_s_theta': 0.2, 'min_sigma_s': 0.005,
    'min_sigma_theta': 0.05, 'sigma_r0': 0.02, 'k_r': 0.03, 'sigma_b_deg': 2.0,
    'gate': 9.21,
    'tag_sigma': 0.02,                # m, 1-sigma of each tape-measured tag position
    'start_sigma': [0.03, 0.03, 0.05],  # m, m, rad: how well the start pose was placed
    'log_dir': os.path.expanduser('~/hw4_logs'),
}

TRAJ_HEADER = ['t', 'x', 'y', 'theta', 'sd_x', 'sd_y', 'sd_theta', 'waypoint']
OBS_HEADER = ['t', 'waypoint', 'id', 'range_m', 'bearing_deg', 'robot_x', 'robot_y',
              'robot_theta_deg', 'result', 'mahalanobis']


def load_known_map(ekf, start, start_sigma, tags, tag_sigma):
    """
    Start the EKF at the measured start pose and put every tag in the state at
    its measured position. The tags get a small uncertainty and no correlation
    with the robot, so sightings mostly correct the robot, not the tags.
    """
    ids = sorted(tags)
    n = 3 + 2 * len(ids)
    s = np.zeros(n)
    s[:3] = start
    S = np.zeros((n, n))
    S[:3, :3] = np.diag(np.square(start_sigma))
    for k, tid in enumerate(ids):
        j = 3 + 2 * k
        s[j:j + 2] = tags[tid][:2]
        S[j:j + 2, j:j + 2] = np.eye(2) * tag_sigma ** 2
    ekf.s, ekf.Sigma, ekf.ids = s, S, ids


class Hw4Navigator(Node):
    def __init__(self):
        super().__init__('hw4_navigator', allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)
        given = {n: prm.value for n, prm in self.get_parameters_by_prefix('').items()
                 if n != 'use_sim_time'}
        self.p = {**DEFAULTS, **given}
        for k in ('arena_size', 'obstacles', 'start', 'goal', 'tag_ids', 'tag_x', 'tag_y',
                  'tag_yaw', 'tag_size', 'robot_radius', 'safety_margin'):
            if k not in self.p:
                raise RuntimeError(f'parameter "{k}" missing: run with --params-file hw4_params.yaml')
        self.setup()

        self.motor_pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(CompressedImage, self.p['image_topic'], self._img_cb, qos)
        self.create_subscription(CameraInfo, self.p['camera_info_topic'], self._info_cb, 10)
        self._frame = None
        self.K = self.D = None
        self.detector = TagDetector(float(self.p['tag_size']), self.p['tag_family'])

    def setup(self):
        """Everything that doesn't need ROS: world, plan, filter, logs."""
        p = self.p
        self.world = World(p)
        self.planner = str(p['planner'])
        self.plan = plan(self.world, self.planner)
        self.waypoints = list(self.plan.waypoints[1:])      # [0] is the start pose
        self.cam_offset = (float(p['camera_offset_x']), float(p['camera_offset_y']))
        self.ekf = make_ekf(p)
        load_known_map(self.ekf, self.world.start, [float(v) for v in p['start_sigma']],
                       self.world.tags, float(p['tag_sigma']))
        self.out = os.path.join(os.path.expanduser(p['log_dir']),
                                time.strftime(f'hw4_{self.planner}_%Y%m%d_%H%M%S'))
        self.t0 = time.time()
        self.wp_index = 0
        self.wp_reached = False
        self.replans = 0
        self.dist_traveled = 0.0
        self.angle_turned = 0.0
        self.min_clearance = float('inf')
        self.events, self.traj, self.sightings, self.wp_results = [], [], [], []
        self.plans = [(0.0, 'initial', self.plan.points)]
        self.log(f'{self.planner} plan: {len(self.waypoints)} waypoints, '
                 f'{self.plan.metrics["length_m"]:.2f} m, min clearance '
                 f'{self.plan.metrics["min_body_clearance_m"]:.2f} m (robot edge)')
        for x, y, th in self.plan.waypoints:
            self.log(f'    ({x:.3f}, {y:.3f}, {math.degrees(th):+.0f} deg)')

    # ---------------- small helpers ----------------
    def log(self, msg):
        self.get_logger().info(msg)

    def t(self):
        return time.time() - self.t0

    def record(self):
        x, y, th = self.ekf.pose
        sd = np.sqrt(np.maximum(np.diag(self.ekf.Sigma)[:3], 0.0))
        self.traj.append([self.t(), x, y, th, *sd, self.wp_index])
        self.min_clearance = min(self.min_clearance, float(self.world.clearance(x, y)))

    # ---------------- ROS callbacks ----------------
    def _img_cb(self, msg):
        self._frame = (time.time(), bytes(msg.data))

    def _info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            d = np.array(msg.d, dtype=np.float64)
            self.D = np.zeros(5) if (self.p['image_is_rectified'] or d.size == 0) else d

    # ---------------- motors (same as hw3_slam) ----------------
    def _publish(self, left, right):
        m = Float32MultiArray()
        m.data = [float(left), float(right)]
        self.motor_pub.publish(m)

    def stop(self):
        for _ in range(3):
            self._publish(0.0, 0.0)
            time.sleep(0.05)

    def run_for(self, left, right, duration):
        t_end = time.time() + duration
        while time.time() < t_end and rclpy.ok():
            self._publish(left, right)
            time.sleep(0.02)
        self.stop()

    def predict(self, ds, dth):
        self.ekf.predict(ds, dth)
        self.events.append([self.t(), 'predict', ds, dth, '', '', ''])

    def turn(self, angle):
        """Timed turn in place by angle (rad, + = left)."""
        p = self.p
        cmd = p['turn_cmd']
        left, right = (-cmd, cmd) if angle > 0 else (cmd, -cmd)
        rate = p['turn_rate_left' if angle > 0 else 'turn_rate_right'] or p['turn_rate']
        self.run_for(left, right, abs(angle) / rate)
        time.sleep(p['settle_time'])
        self.predict(0.0, angle)
        self.angle_turned += abs(angle)

    def drive(self, duration, backward=False):
        p = self.p
        duration = max(duration, p['min_step_time'])
        sgn = -1.0 if backward else 1.0
        self.run_for(sgn * p['drive_left'], sgn * p['drive_right'], duration)
        time.sleep(p['settle_time'])
        d = sgn * p['drive_speed'] * duration
        self.predict(d, math.radians(p['drive_curve_deg_per_m']) * d)
        self.dist_traveled += abs(d)

    # ---------------- sensing ----------------
    def fresh_frame(self, after, timeout=1.5):
        t_end = time.time() + timeout
        while time.time() < t_end and rclpy.ok():
            f = self._frame
            if f is not None and f[0] > after:
                img = cv2.imdecode(np.frombuffer(f[1], np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            time.sleep(0.02)
        return None

    def detect_tags(self):
        """[(id, range, bearing)] from one fresh frame, robot-center frame. None = no frame."""
        img = self.fresh_frame(after=time.time())
        if img is None:
            return None
        out = []
        for d in self.detector.detect(img, self.K, self.D):
            if d['dist'] > self.p['detect_max_dist'] or \
                    d['candidates'][0][2] > self.p['max_reproj_err']:
                continue
            fx, ly, _ = tag_in_robot_frame(d['candidates'][0][0], d['candidates'][0][1],
                                           self.cam_offset)
            out.append((d['id'], *detection_to_range_bearing(fx, ly)))
        return out

    def look(self):
        """Correct the pose with every known tag in view. Returns how many were used."""
        dets = self.detect_tags()
        if dets is None:
            self.get_logger().warn('No camera frame received')
            self.record()
            return 0
        used, msgs = 0, []
        for tid, r, b in dets:
            pose = self.ekf.pose
            if tid not in self.ekf.ids:
                result, m = 'unknown', None          # not in the map: ignore
            else:
                result, m = self.ekf.observe(tid, r, b)
                used += result == 'updated'
            self.events.append([self.t(), 'obs', tid, r, b, result, '' if m is None else m])
            self.sightings.append([self.t(), self.wp_index, tid, r, math.degrees(b),
                                   pose[0], pose[1], math.degrees(pose[2]), result,
                                   '' if m is None else m])
            msgs.append(f'{tid}:{result}' + ('' if m is None else f'(M={m:.1f})'))
        x, y, th = self.ekf.pose
        sd = np.sqrt(np.maximum(np.diag(self.ekf.Sigma)[:3], 0.0))
        self.log(f'  tags [{", ".join(msgs) or "none"}] -> pose ({x:.2f}, {y:.2f}, '
                 f'{math.degrees(th):+.0f} deg) +/- {100 * max(sd[0], sd[1]):.0f} cm')
        self.record()
        return used

    # ---------------- safety ----------------
    def min_allowed(self):
        """Closest the robot CENTER may get to an obstacle while driving."""
        return float(self.world.robot_radius) + 0.5 * float(self.p['safety_margin'])

    def path_blocked(self, wp):
        x, y, _ = self.ekf.pose
        return self.world.segment_clearance((x, y), wp[:2]) < self.min_allowed()

    def replan(self, why):
        """Re-plan from the current estimate with the same planner. True if it worked."""
        if not self.p['replan'] or self.replans >= self.p['max_replans']:
            return False
        x, y, th = self.ekf.pose
        q = dict(self.p, start=[x, y, th])
        try:
            new = plan(World(q), self.planner)
        except (ValueError, RuntimeError) as e:
            self.get_logger().warn(f'  re-plan failed ({e}); keeping the current plan')
            return False
        self.replans += 1
        self.waypoints = list(new.waypoints[1:])
        self.plans.append((self.t(), why, new.points))
        self.log(f'  RE-PLAN ({why}): {len(self.waypoints)} waypoints, '
                 f'{path_length(new.points):.2f} m')
        return True

    # ---------------- waypoint following (HW3 logic) ----------------
    def turn_step(self, err, why, prefix=''):
        cap = math.radians(self.p['max_turn_step_deg'])
        angle = math.copysign(cap, err) if cap > 0 and abs(err) > cap else err
        self.log(f'  {prefix}turn {math.degrees(angle):+.0f} deg {why}')
        self.turn(angle)

    def step_toward(self, wp, last):
        """One motion toward wp. Returns True when wp is done (no motion made)."""
        x, y, th = self.ekf.pose
        dx, dy = wp[0] - x, wp[1] - y
        dist = math.hypot(dx, dy)
        p = self.p
        along = dx * math.cos(th) + dy * math.sin(th)
        tol = p['goal_tol'] if last else p['reach_radius']
        if dist <= p['pos_tol'] or (dist <= tol and abs(along) <= p['pos_tol']):
            if not self.wp_reached:
                self.log(f'  reached (within {dist * 100:.0f} cm)')
            self.wp_reached = True
        if not self.wp_reached:
            err = wrap(math.atan2(dy, dx) - th)
            side = dist * abs(math.sin(err))
            if dist < 0.35 and abs(err) > math.radians(p['back_up_deg']) and side <= tol:
                back = -along
                self.log(f'  back up {back:.2f} m')
                self.drive(min(p['step_time'], back / p['drive_speed']), backward=True)
            elif (abs(err) > p['heading_tol']
                  and not (abs(err) < math.pi / 2 and side < 0.8 * p['pos_tol'])):
                self.turn_step(err, 'to face waypoint')
            else:
                ahead = dist * math.cos(err)
                t = min(p['step_time'], ahead / p['drive_speed'])
                d = t * p['drive_speed']
                end = (x + d * math.cos(th), y + d * math.sin(th))
                if self.world.segment_clearance((x, y), end) < self.min_allowed():
                    # this step would graze the obstacle: re-plan, or refuse to drive
                    if not self.replan('step too close to an obstacle'):
                        self.get_logger().error('  step too close to an obstacle and no '
                                                're-plan possible; stopping')
                        raise RuntimeError('blocked')
                    return False
                self.log(f'  drive {d:.2f} m ({dist:.2f} m left)')
                self.drive(t)
            return False
        if last and self.world.goal_theta is None:
            return True                       # the goal has no required heading
        err = wrap(wp[2] - th)
        if abs(err) <= p['ang_tol']:
            return True
        self.turn_step(err, 'to the next leg\'s heading', prefix='at waypoint; ')
        return False

    # ---------------- main loop ----------------
    def wait_for_inputs(self, timeout=15.0):
        t_end = time.time() + timeout
        while time.time() < t_end and rclpy.ok():
            if self.K is not None and self._frame is not None:
                break
            time.sleep(0.1)
        if self.K is None:
            raise RuntimeError('No camera_info. Is the camera running with a calibration YAML?')
        if self._frame is None:
            raise RuntimeError('No images on the camera topic.')
        time.sleep(0.5)

    def run(self):
        self.wait_for_inputs()
        self.log('Initial look at the start pose')
        self.t0 = time.time()
        self.look()
        k = 0
        while k < len(self.waypoints) and rclpy.ok():
            wp = self.waypoints[k]
            last = k == len(self.waypoints) - 1
            self.wp_index += 1
            self.wp_reached = False
            self.log(f'=== waypoint {k + 1}/{len(self.waypoints)}: ({wp[0]:.2f}, {wp[1]:.2f}, '
                     f'{math.degrees(wp[2]):+.0f} deg){" GOAL" if last else ""} ===')
            if self.path_blocked(wp) and self.replan('straight line to the waypoint is blocked'):
                k = 0
                continue
            n_plans = len(self.plans)
            for _ in range(int(self.p['max_steps_per_waypoint'])):
                if not rclpy.ok() or self.step_toward(wp, last):
                    break
                if len(self.plans) != n_plans:
                    break                     # re-planned inside step_toward
                self.look()
            else:
                self.get_logger().warn('  step limit reached; moving on')
            x, y, th = self.ekf.pose
            self.wp_results.append([self.wp_index, *wp, x, y, th, self.t()])
            if len(self.plans) != n_plans:
                k = 0                         # follow the new plan from its start
            else:
                k += 1
        x, y, _ = self.ekf.pose
        gx, gy = self.world.goal
        self.log(f'Done in {self.t():.0f} s: {self.dist_traveled:.2f} m driven, '
                 f'{math.degrees(self.angle_turned):.0f} deg turned, {self.replans} re-plans. '
                 f'Estimated distance to goal {100 * math.hypot(gx - x, gy - y):.0f} cm.')

    # ---------------- logs ----------------
    def save_logs(self):
        out = self.out
        os.makedirs(out, exist_ok=True)
        write_csv(out, 'events.csv', ['t', 'kind', 'a', 'b', 'c', 'd', 'e'], self.events)
        write_csv(out, 'trajectory.csv', TRAJ_HEADER, self.traj)
        write_csv(out, 'observations.csv', OBS_HEADER, self.sightings)
        write_csv(out, 'waypoints.csv', ['waypoint', 'goal_x', 'goal_y', 'goal_theta',
                                         'est_x', 'est_y', 'est_theta', 't'], self.wp_results)
        write_csv(out, 'plan.csv', ['plan', 't', 'why', 'x', 'y'],
                  [[i, round(t, 2), why, float(px), float(py)]
                   for i, (t, why, pts) in enumerate(self.plans) for px, py in pts])
        with open(os.path.join(out, 'params.json'), 'w') as f:
            json.dump({k: (list(v) if hasattr(v, '__len__') and not isinstance(v, str) else v)
                       for k, v in self.p.items()}, f, indent=2)
        x, y, th = self.ekf.pose
        gx, gy = self.world.goal
        m = self.plan.metrics
        used = sum(1 for s in self.sightings if s[8] == 'updated')
        lines = [
            f'HW4 run {os.path.basename(out)}  planner: {self.planner}',
            f'planned: {m["length_m"]:.2f} m, {m["n_waypoints"]} waypoints, min clearance '
            f'{m["min_body_clearance_m"]:.2f} m (robot edge), est. {m["est_time_s"]:.0f} s',
            f'driven:  {self.dist_traveled:.2f} m, turned {math.degrees(self.angle_turned):.0f} deg, '
            f'{self.t():.0f} s, {self.replans} re-plans',
            f'closest estimated approach to an obstacle: '
            f'{self.min_clearance - float(self.world.robot_radius):.2f} m (robot edge)',
            f'final pose estimate ({x:.3f}, {y:.3f}, {math.degrees(th):+.1f} deg), '
            f'{100 * math.hypot(gx - x, gy - y):.1f} cm from the goal',
            f'tag sightings: {len(self.sightings)}, used {used}',
        ]
        with open(os.path.join(out, 'summary.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        self.log(f'Saved logs to {out}')


def main(args=None):
    rclpy.init(args=args)
    node = Hw4Navigator()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        node.run()
    except KeyboardInterrupt:
        node.log('Interrupted')
    except Exception as e:
        node.get_logger().error(f'Stopped: {e}')
    finally:
        try:
            node.stop()
        except Exception:
            pass
        node.save_logs()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
