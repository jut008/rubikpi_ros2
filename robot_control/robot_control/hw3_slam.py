#!/usr/bin/env python3
"""
CSE 276A HW3: EKF SLAM with AprilTags while driving a square / octagon.

Same stop-and-go loop as HW2, but there is NO tag map. The robot starts at
(0, 0, 0) in its own frame and builds the map as it goes:
    1. Plan:    pick one motion toward the current waypoint using the EKF pose.
    2. Act:     run it, then stop.
    3. Predict: EKF prediction from the odometry (ds, dtheta).
    4. Look:    grab a fresh frame, detect tags, and for each tag
                - new ID   -> add it to the state (augment)
                - known ID -> EKF update (after the Mahalanobis gate)
    5. Log:     robot pose + uncertainty every step, every raw odometry step
                and tag sighting (so replay_slam.py can re-run the filter
                offline), and a map snapshot after every loop.

Needs (in other terminals):
    ros2 launch robot_vision_camera robot_vision_camera.launch.py
    ros2 run robot_control motor_control
Then:
    ros2 run robot_control hw3_slam --ros-args \
        --params-file ~/ros2_ws/src/rubikpi_ros2/robot_control/config/hw3_params.yaml
"""
import csv
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
from sensor_msgs.msg import CameraInfo, CompressedImage, Imu
from std_msgs.msg import Float32MultiArray

from robot_control.ekf_slam import (detection_to_range_bearing, make_ekf, octagon_path,
                                    square_path, wrap)
from robot_control.tag_localization import TagDetector, tag_in_robot_frame


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Hw3Slam(Node):
    def __init__(self):
        super().__init__('hw3_slam')
        P = self.declare_parameter
        # --- path ---
        P('paths', 'square')             # 'square', 'octagon', or 'square,octagon'
        P('square_size', 1.0)            # n, meters
        P('octagon_side', 0.6)           # m, meters
        P('loops', 1)                    # times around each path
        # --- tags / camera ---
        P('tag_size', 0.163)
        P('tag_family', '36h11')
        P('max_tag_dist', 3.0)           # ignore tags farther than this (m)
        P('max_reproj_err', 3.0)         # px; ignore poorly fit detections
        P('image_topic', '/camera/image_raw/compressed')
        P('camera_info_topic', '/camera/camera_info')
        P('camera_offset_x', 0.10)
        P('camera_offset_y', 0.0)
        P('image_is_rectified', False)
        # --- motion (HW1/HW2 calibration) ---
        P('drive_left', 0.32)
        P('drive_right', 0.40)
        P('drive_speed', 0.495)
        P('turn_cmd', 0.3)
        P('turn_rate', 2.051)
        P('turn_rate_left', 1.963)
        P('turn_rate_right', 2.138)
        P('turn_lead', 0.05)
        P('search_cmd', 0.3)
        P('search_step', 0.35)
        P('imu_topic', '/imu_data')
        P('use_imu', False)
        P('imu_yaw_sign', 1.0)
        # --- control loop ---
        P('step_time', 0.5)              # max seconds of driving per step
        P('min_step_time', 0.10)
        P('settle_time', 0.4)
        P('pos_tol', 0.05)
        P('reach_radius', 0.15)
        P('ang_tol', 0.10)
        P('heading_tol', 0.12)
        P('max_blind_steps', 3)          # steps without a known tag before searching
        P('max_steps_per_waypoint', 40)
        P('log_dir', os.path.expanduser('~/hw3_logs'))
        # --- EKF noise (see ekf_slam.EKFSlam) ---
        P('k_s', 0.05)
        P('k_theta', 0.10)
        P('k_s_theta', 0.05)
        P('min_sigma_s', 0.005)
        P('min_sigma_theta', 0.005)
        P('sigma_r0', 0.02)
        P('k_r', 0.03)
        P('sigma_b_deg', 2.0)
        P('gate', 9.21)

        self.p = {n: prm.value for n, prm in self.get_parameters_by_prefix('').items()
                  if n != 'use_sim_time'}
        self.cam_offset = (self.p['camera_offset_x'], self.p['camera_offset_y'])
        self.detector = TagDetector(self.p['tag_size'], self.p['tag_family'])
        self.ekf = make_ekf(self.p)

        # waypoints, each tagged with (path name, loop number, last-of-loop?)
        self.plan = []
        for name in [s.strip() for s in self.p['paths'].split(',') if s.strip()]:
            if name == 'square':
                one = square_path(self.p['square_size'], 1)[1:]
            elif name == 'octagon':
                one = octagon_path(self.p['octagon_side'], 1)[1:]
            else:
                raise ValueError(f'Unknown path "{name}" (use square / octagon)')
            for loop in range(1, self.p['loops'] + 1):
                for k, wp in enumerate(one):
                    self.plan.append((wp, name, loop, k == len(one) - 1))

        # --- ROS I/O ---
        self.motor_pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        img_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(CompressedImage, self.p['image_topic'], self._img_cb, img_qos)
        self.create_subscription(CameraInfo, self.p['camera_info_topic'], self._info_cb, 10)
        self.create_subscription(Imu, self.p['imu_topic'], self._imu_cb, 10)
        self._frame = None
        self.K = None
        self.D = None
        self.imu_yaw = None
        self._imu_raw_prev = None

        # --- logs ---
        self.t0 = time.time()
        self.wp_index = 0
        self.blind_steps = 0
        self.dist_traveled = 0.0
        self.angle_turned = 0.0
        self.events = []      # t, kind, a, b, c, d, e (see save_logs)
        self.traj = []        # t, x, y, theta, sd_x, sd_y, sd_theta, n_landmarks, wp
        self.loop_maps = []   # path, loop, id, x, y, sxx, sxy, syy
        self.wp_results = []
        self.get_logger().info(f'{len(self.plan)} waypoints: '
                               f'{[tuple(round(v, 2) for v in w[0]) for w in self.plan]}')

    # ---------------- callbacks ----------------
    def _img_cb(self, msg):
        self._frame = (time.time(), bytes(msg.data))

    def _info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            d = np.array(msg.d, dtype=np.float64)
            self.D = np.zeros(5) if (self.p['image_is_rectified'] or d.size == 0) else d

    def _imu_cb(self, msg):
        raw = self.p['imu_yaw_sign'] * yaw_from_quaternion(msg.orientation)
        if self._imu_raw_prev is None:
            self.imu_yaw = raw
        else:
            self.imu_yaw += wrap(raw - self._imu_raw_prev)
        self._imu_raw_prev = raw

    # ---------------- motors ----------------
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

    def imu_ok(self):
        return self.p['use_imu'] and self.imu_yaw is not None

    def t(self):
        return time.time() - self.t0

    def predict(self, ds, dth):
        """EKF prediction + log the raw odometry so the run can be replayed."""
        self.ekf.predict(ds, dth)
        self.events.append([self.t(), 'predict', ds, dth, '', '', ''])

    def turn(self, angle, cmd=None):
        """Turn in place by angle (rad, + = left). Returns the measured change."""
        cmd = cmd if cmd is not None else self.p['turn_cmd']
        left, right = (-cmd, cmd) if angle > 0 else (cmd, -cmd)
        if self.imu_ok():
            yaw0 = self.imu_yaw
            target = max(0.0, abs(angle) - self.p['turn_lead'])
            t_end = time.time() + abs(angle) / self.p['turn_rate'] * 2.0 + 1.0
            while abs(self.imu_yaw - yaw0) < target and time.time() < t_end and rclpy.ok():
                self._publish(left, right)
                time.sleep(0.02)
            self.stop()
            time.sleep(self.p['settle_time'])
            turned = self.imu_yaw - yaw0
        else:
            rate = self.p['turn_rate_left' if angle > 0 else 'turn_rate_right'] \
                or self.p['turn_rate']
            self.run_for(left, right, abs(angle) / rate)
            time.sleep(self.p['settle_time'])
            turned = angle
        self.predict(0.0, turned)
        self.angle_turned += abs(turned)
        return turned

    def drive(self, duration, backward=False):
        duration = max(duration, self.p['min_step_time'])
        sgn = -1.0 if backward else 1.0
        yaw0 = self.imu_yaw if self.imu_ok() else None
        self.run_for(sgn * self.p['drive_left'], sgn * self.p['drive_right'], duration)
        time.sleep(self.p['settle_time'])
        dth = (self.imu_yaw - yaw0) if yaw0 is not None else 0.0
        d = sgn * self.p['drive_speed'] * duration
        self.predict(d, dth)
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

    def look(self):
        """
        Detect tags in a fresh frame and feed each one to the EKF.
        Returns the number of KNOWN landmarks that corrected the estimate.
        """
        img = self.fresh_frame(after=time.time())
        if img is None:
            self.get_logger().warn('No camera frame received')
            self.record()
            return 0
        n_upd = 0
        msgs = []
        for d in self.detector.detect(img, self.K, self.D):
            if d['dist'] > self.p['max_tag_dist'] or d['candidates'][0][2] > self.p['max_reproj_err']:
                continue
            rvec, tvec, _ = d['candidates'][0]
            fx, ly, _ = tag_in_robot_frame(rvec, tvec, self.cam_offset)
            r, b = detection_to_range_bearing(fx, ly)
            result, m = self.ekf.observe(d['id'], r, b)
            self.events.append([self.t(), 'obs', d['id'], r, b, result,
                                '' if m is None else m])
            n_upd += result == 'updated'
            msgs.append(f'{d["id"]}:{result}' + ('' if m is None else f'(M={m:.1f})'))
        x, y, th = self.ekf.pose
        self.get_logger().info(
            f'  tags [{", ".join(msgs) or "none"}] -> pose ({x:+.2f}, {y:+.2f}, '
            f'{math.degrees(th):+.0f} deg), {self.ekf.n_landmarks} landmarks')
        self.blind_steps = 0 if n_upd else self.blind_steps + 1
        self.record()
        return n_upd

    def search(self):
        """Turn in small steps until a known landmark is seen (max one turn)."""
        self.get_logger().warn('No known tag for a while: searching')
        turned = 0.0
        while turned < 2 * math.pi and rclpy.ok():
            turned += abs(self.turn(-self.p['search_step'], cmd=self.p['search_cmd']))
            if self.look():
                return True
        return False

    def record(self):
        x, y, th = self.ekf.pose
        sd = np.sqrt(np.maximum(np.diag(self.ekf.Sigma)[:3], 0.0))
        self.traj.append([self.t(), x, y, th, *sd, self.ekf.n_landmarks, self.wp_index])

    def snapshot_map(self, path, loop):
        for tid, (x, y, S) in self.ekf.landmarks().items():
            self.loop_maps.append([path, loop, tid, x, y, S[0, 0], S[0, 1], S[1, 1]])

    # ---------------- control (same logic as HW2) ----------------
    def step_toward(self, wp):
        x, y, th = self.ekf.pose
        dx, dy = wp[0] - x, wp[1] - y
        dist = math.hypot(dx, dy)
        p = self.p
        along = dx * math.cos(th) + dy * math.sin(th)
        sideways_only = dist <= p['reach_radius'] and abs(along) <= p['pos_tol']
        if dist > p['pos_tol'] and not sideways_only:
            err = wrap(math.atan2(dy, dx) - th)
            if dist < 0.35 and abs(err) > 2.0:
                back = -along
                self.get_logger().info(f'  back up {back:.2f} m')
                self.drive(min(p['step_time'], back / p['drive_speed']), backward=True)
            elif (abs(err) > p['heading_tol']
                  and not (abs(err) < math.pi / 2
                           and dist * abs(math.sin(err)) < 0.8 * p['pos_tol'])):
                self.get_logger().info(f'  turn {math.degrees(err):+.0f} deg to face waypoint')
                self.turn(err)
            else:
                ahead = dist * math.cos(err)
                t = min(p['step_time'], ahead / p['drive_speed'])
                self.get_logger().info(f'  drive {t * p["drive_speed"]:.2f} m ({dist:.2f} m left)')
                self.drive(t)
            return False
        err = wrap(wp[2] - th)
        if abs(err) <= p['ang_tol']:
            return True
        self.get_logger().info(f'  at waypoint; turn {math.degrees(err):+.0f} deg to final heading')
        self.turn(err)
        return False

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
        if self.p['use_imu'] and self.imu_yaw is None:
            self.get_logger().warn('No IMU data; turns will be timed instead of measured')

    def run(self):
        self.wait_for_inputs()
        self.get_logger().info('Initial look (robot pose defines the map frame: (0, 0, 0))')
        self.look()
        for i, (wp, path, loop, last) in enumerate(self.plan, start=1):
            self.wp_index = i
            self.get_logger().info(f'=== {path} loop {loop}, waypoint {i}: '
                                   f'({wp[0]:.2f}, {wp[1]:.2f}, {math.degrees(wp[2]):.0f} deg) ===')
            for _ in range(self.p['max_steps_per_waypoint']):
                if not rclpy.ok():
                    return
                if self.step_toward(wp):
                    break
                if (not self.look() and self.ekf.n_landmarks
                        and self.blind_steps > self.p['max_blind_steps']):
                    if not self.search():
                        self.get_logger().warn('Nothing found after a full turn; continuing')
                        self.blind_steps = 0
            x, y, th = self.ekf.pose
            self.wp_results.append([i, path, loop, *wp, x, y, th, self.t()])
            if last:
                self.snapshot_map(path, loop)
                self.events.append([self.t(), 'loop_end', path, loop, '', '', ''])
                self.get_logger().info(f'--- {path} loop {loop} done: '
                                       f'{self.ekf.n_landmarks} landmarks ---')
        self.get_logger().info(
            f'Done. {self.ekf.n_landmarks} landmarks. Distance {self.dist_traveled:.2f} m, '
            f'rotation {math.degrees(self.angle_turned):.0f} deg')

    # ---------------- logs ----------------
    def save_logs(self):
        out = os.path.join(self.p['log_dir'], time.strftime('hw3_%Y%m%d_%H%M%S'))
        os.makedirs(out, exist_ok=True)

        def write(name, header, rows):
            with open(os.path.join(out, name), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)

        # events: predict rows = (ds, dtheta); obs rows = (id, range, bearing, result, M)
        write('events.csv', ['t', 'kind', 'a', 'b', 'c', 'd', 'e'], self.events)
        write('trajectory.csv', ['t', 'x', 'y', 'theta', 'sd_x', 'sd_y', 'sd_theta',
                                 'n_landmarks', 'waypoint'], self.traj)
        write('map_loops.csv', ['path', 'loop', 'id', 'x', 'y', 'sxx', 'sxy', 'syy'],
              self.loop_maps)
        write('map_final.csv', ['id', 'x', 'y', 'sxx', 'sxy', 'syy'],
              [[tid, x, y, S[0, 0], S[0, 1], S[1, 1]]
               for tid, (x, y, S) in self.ekf.landmarks().items()])
        write('waypoints.csv', ['waypoint', 'path', 'loop', 'goal_x', 'goal_y', 'goal_theta',
                                'est_x', 'est_y', 'est_theta', 't'], self.wp_results)
        with open(os.path.join(out, 'params.json'), 'w') as f:
            json.dump(self.p, f, indent=2)
        self.get_logger().info(f'Saved logs to {out}')


def main(args=None):
    rclpy.init(args=args)
    node = Hw3Slam()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted')
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
