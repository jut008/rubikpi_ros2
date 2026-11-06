# CSE 276A HW3: EKF SLAM, session notes (Nov 5, 2026)

## The assignment in plain terms

- **HW2:** we gave the robot a map of where the tags were, and it used that map to figure out where it was.
- **HW3:** no map. The robot starts at (0, 0, 0) in its own frame and figures out **where the tags are and where it is at the same time** (SLAM).
- The Kalman filter keeps one list of guesses plus an uncertainty for each:
  `[robot x, robot y, robot θ, tag A x, tag A y, tag B x, tag B y, ...]`
- Every step:
  1. **Move (predict):** shift the robot guess using odometry and get a little less sure.
  2. **See a known tag (update):** compare where we expected it with where we saw it, then nudge the robot and tag guesses.
  3. **See a new tag ID (augment):** add it to the list. That's how the filter handles not knowing how many tags there are.
- Tags aren't used to decide where to go. The square and octagon are just waypoints. Tags only help the robot figure out where it is.
- At the end we compare the robot's map with the real tag positions we measure with a tape measure (part 4).

### Assignment checklist
- [ ] Part 1: tape out a 10×10 ft area (8 ft minimum), 8 tags (2 per side, different IDs), measure tag positions, sketch the ground-truth map
- [x] Part 2: EKF SLAM implemented (`ekf_slam.py`, `hw3_slam.py`)
- [ ] Part 3: run the square path, then the octagon path
- [ ] Part 4: measure the start pose, move the map into the world frame, compute the average landmark error, compare 1 loop vs several loops

## Lectures that matter

| Lecture | Use it for |
|---|---|
| 8a: The Kalman Filter | predict/update equations, EKF Jacobians, Mahalanobis gate |
| 8b: SLAM | state = robot pose + landmarks, "augment state vector" for new landmarks |
| 9: Localization | odometry motion model and uncertainty propagation (predict step) |
| 3: Reference frames | transforming the map into the world frame (part 4) |

Not needed: 7 (visual servoing), 10 (graph SLAM), 12 (planning). Lecture 11 had no slides.

Not spelled out in the slides (we used standard versions; check the whiteboard or recording if he showed his own):
- the landmark measurement model (we use range + bearing to the tag center)
- the starting covariance for a new landmark
- how to tune Q and R

## Connecting to the robot

The hotspot gives the Mac **IPv6 only** (`ifconfig` shows `192.0.0.2`, no `172.20.10.x`), so the IPv4 address doesn't work. Use the robot's IPv6 link-local address.

1. Turn on the hotspot (keep the Personal Hotspot screen open), power on the robot, and put the Mac on the hotspot.
2. Try the last known address:
   ```
   ssh ubuntu@fe80::1213:9a02:4e81:60b4%en0
   ```
3. If it hangs for more than about 10 s, get the new address over USB serial:
   ```
   ls /dev/cu.usbmodem*
   screen /dev/cu.usbmodem58C60626701 115200
   ```
   Log in, then:
   ```
   stty cols 200
   ip -6 addr show wlan0 | grep fe80
   ```
   Exit screen (Ctrl+A, K, Y) and SSH to `ubuntu@<fe80 address>%en0`.
4. `scp` needs brackets and quotes:
   ```
   scp file 'ubuntu@[fe80::1213:9a02:4e81:60b4%en0]:~/'
   ```

Fixes we hit:
- `screen` "bind ... Permission denied" → `sudo rm -rf ~/.screen`
- `iw` isn't installed on the robot (skip the power_save step, or use the `nmcli ... 802-11-wireless.powersave 2` version)
- Camera `VendorTagDescriptor` errors are harmless driver noise. Stop the "Broadcast message" spam with:
  ```
  sudo mkdir -p /etc/systemd/journald.conf.d
  echo -e "[Journal]\nForwardToWall=no" | sudo tee /etc/systemd/journald.conf.d/nowall.conf
  sudo systemctl restart systemd-journald
  ```

## Robot health check (all passed)
- Code on the robot matched GitHub (`55bd596`, HW2 commit)
- Camera: `GStreamer pipeline started`, about 10 fps on `/camera/image_raw/compressed`
- Motors: serial link on `/dev/ttyUSB0`, IMU publishing (the "safety stop" warning is normal)

## What we added (branch `hw3`, HW2 code untouched)

| File | What it does |
|---|---|
| `robot_control/ekf_slam.py` | the filter (no ROS): predict, update, add landmark, Mahalanobis gate, square/octagon waypoints |
| `robot_control/hw3_slam.py` | ROS node: HW2's stop-and-go driving with the EKF in place of the known map; logs everything |
| `robot_control/replay_slam.py` | reruns a saved run on a laptop with different noise settings |
| `robot_control/analyze_map.py` | plots the map; with ground truth, computes errors for part 4 |
| `config/hw3_params.yaml` | path, sizes, loops, motion calibration (copied from HW2), EKF noise |
| `test/test_ekf_slam.py` | simulation test (fake robot, 8 tags) |
| `setup.py` | registers `hw3_slam` |

Each run saves a folder `~/hw3_logs/hw3_<date>_<time>/` with `events.csv` (raw odometry and sightings), `trajectory.csv`, `map_final.csv`, `map_loops.csv` (map after each loop), `waypoints.csv`, and `params.json`.

## How to run

tmux windows on the robot (Ctrl+B then C = new window, Ctrl+B then a number = switch, Ctrl+B then D = detach; `tmux attach` to come back):

```
# window 1: camera
cd ~/ros2_ws && source install/setup.bash
ros2 launch robot_vision_camera robot_vision_camera.launch.py

# window 2: motors
cd ~/ros2_ws && source install/setup.bash
ros2 run robot_control motor_control

# window 3: SLAM (unplug the USB cable first)
cd ~/ros2_ws && source install/setup.bash
ros2 run robot_control hw3_slam --ros-args \
  --params-file ~/ros2_ws/src/rubikpi_ros2/robot_control/config/hw3_params.yaml \
  -p square_size:=0.5
```

Useful overrides: `-p loops:=2`, `-p paths:=octagon -p octagon_side:=0.4`, `-p paths:=square,octagon`.
Both paths start at the robot's spot and turn **left**, so leave room ahead of it and to its left.

Rebuild after code changes:
```
cd ~/ros2_ws && colcon build --packages-select robot_control && source install/setup.bash
```

Offline checks:
```
python3 test/test_ekf_slam.py                        # simulation test
python3 robot_control/replay_slam.py LOG_DIR --k_theta 0.15   # retune without driving
python3 robot_control/analyze_map.py LOG_DIR         # picture of the map -> slam_map.png
```

Part 4, once tags and start pose are measured (world frame, e.g. origin at one corner):
```
python3 robot_control/analyze_map.py LOG_DIR --gt ground_truth.csv \
  --start X0 Y0 THETA0_DEG --units ft --room 10
python3 robot_control/analyze_map.py --gt ground_truth.csv --start X0 Y0 THETA0_DEG \
  --units ft --room 10 --sketch                      # part 1 sketch
```
`ground_truth.csv` format: `id,x,y`, one line per tag.

## Results so far

- **Simulation test** (matched on laptop and robot): square, 6.3 cm mean map error (odometry alone drifted 38.5 cm); octagon, 4.8 cm (odometry alone 61.2 cm).
- **First real run** (cross setup: 4 tags in front, left, behind, right; 0.5 m square, 1 loop; log `hw3_20261106_003733`):
  - found all 4 tags, every re-sighting passed the gate (M between 0.0 and 2.4)
  - on each side the robot only saw the tag it had just added, so drift went unnoticed until it re-saw tag 1 at the end (loop closure), which corrected the heading by about **28°**
  - the filter ended thinking it was facing the start direction, about 13 cm to the right of the start. That matches what we saw (facing the start direction, a bit to the right).
  - takeaway: the turns drift a few degrees each. More tags in view (8 tags, 2 per wall) and more loops should help.

## Next steps

1. Apply `hw3-plot-map.patch` on the Mac and robot, plot the first run, and check that the 4 tags form a cross.
2. In the cross setup, try `loops:=2` and the octagon (`octagon_side:=0.4`).
3. Maybe recalibrate the turn rates (each turn seems a few degrees off).
4. Set up the real 10×10 ft area, measure the tags and start pose, and make `ground_truth.csv`.
5. Real runs: square (1 loop, then 2 or 3), octagon (1 loop, then 2 or 3), then run `analyze_map.py` for each.
6. Report: ground-truth sketch, method (cite Lecture 8 notation), map comparison plots, average errors, error per loop, video link.
