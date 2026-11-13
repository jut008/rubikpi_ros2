# HW3 EKF SLAM: run notes

Code: `ekf_slam.py` (filter), `hw3_slam.py` (robot node), `replay_slam.py`
(offline re-run), `analyze_map.py` (plots + error vs ground truth),
`config/hw3_params.yaml`, `test/test_ekf_slam.py` (simulation).

## Before your next run

1. **Get the code on the robot and rebuild**
   ```bash
   cd ~/ros2_ws/src/rubikpi_ros2 && git pull        # on the hw3 branch
   cd ~/ros2_ws && colcon build --packages-select robot_control && source install/setup.bash
   ```
2. **Set up the area**: 8 ft x 8 ft (minimum), two AprilTags per side, each
   with a different ID. Tags at about camera height, facing into the area.
3. **Measure the ground truth** in a world frame you choose (e.g. origin at
   one corner, x along one wall, y along the other). Write `ground_truth.csv`:
   ```
   id,x,y
   0,2.67,0
   ...
   ```
   Measure to the **center** of each tag. Keep the units consistent (ft is fine, pass `--units ft`).
4. **Mark the start pose on the floor** (tape under the robot's center plus
   a line for its heading) and measure it in the same world frame:
   x0, y0, and heading in degrees CCW from world +x. The robot does NOT get
   this; it's only used afterwards to compare maps.
5. **Place the robot.** Default (`center_path: false`): the robot starts at
   a **corner of the path**, drives straight ahead one side, turns left,
   and goes around counter-clockwise back to its start. The path lies
   AHEAD and to the LEFT of the start. To center the path in an 8 ft
   (2.44 m) area, put the robot's center this far from the walls:

   | path | from the right wall | from the back wall | closest to any wall |
   |---|---|---|---|
   | square n = 1.6 m | 0.42 m | 0.42 m | 0.42 m (tight) |
   | square n = 1.2 m | 0.62 m | 0.62 m | 0.62 m |
   | square n = 1.0 m (default) | 0.72 m | 0.72 m | 0.72 m |
   | octagon m = 0.6 m | 0.50 m | 0.92 m (it swings 0.42 m behind the start) | 0.50 m |

   ("right" and "back" = relative to the way the robot faces at the start.)

   Or `-p center_path:=true`: put the robot in the middle; it drives
   straight ahead half a side, turns left, goes around the path centered on
   its start, and comes back to the middle.
6. **Straighten the drive** (do this whenever the battery or floor changes).
   The Nov 13 runs curved LEFT about 12-16 deg per meter on "straight"
   drives; with the filter set to expect straight drives it rejected good
   tags, got lost by the 3rd corner, and drove into a tag. Run, with
   `motor_control` running:
   ```bash
   python3 ~/ros2_ws/src/rubikpi_ros2/robot_control/robot_control/calibrate_motion.py \
       --straight --drive-left 0.32 --drive-right 0.40
   ```
   It drives ~1 m, you type how far forward and how far sideways it ended,
   and it suggests new `drive_left`/`drive_right`; repeat until it says
   "Straight enough", then paste its lines into `hw3_params.yaml`
   (`drive_curve_deg_per_m` = whatever curve is left; 0 if straight).
   The current `drive_curve_deg_per_m: 12` + `k_s_theta: 0.2` already let
   the filter cope with the curve (fit from the logs) if you skip this.

## Running

Three terminals on the robot:
```bash
ros2 launch robot_vision_camera robot_vision_camera.launch.py
ros2 run robot_control motor_control
ros2 run robot_control hw3_slam --ros-args \
    --params-file ~/ros2_ws/src/rubikpi_ros2/robot_control/config/hw3_params.yaml \
    -p paths:=square
```
Then the same with `-p paths:=octagon`. **Do them as separate runs**: the
assignment compares the two maps (`square,octagon` builds one combined map).

Useful overrides: `-p loops:=2` (more loops, for "does driving more help?"),
`-p square_size:=1.0`, `-p octagon_side:=0.5`, `-p center_path:=true`
(start in the middle of the path instead of a corner), `-p return_to_start:=false`.

**Search is off** (`search: false`). It used to spin up to a full circle
whenever no known tag was seen for a few steps, which added turn error and
pulled the robot off the square. For SLAM the robot just keeps following the
path and corrects at the next tag. If you turn it on (`-p search:=true`), it
sweeps at most `search_max_deg` (90) and turns back if it finds nothing.

Suggested runs for the report:
- square, 1 loop and square, 2-3 loops
- octagon, 1 loop and octagon, 2-3 loops

Put the robot back on the same start mark each time.

Ctrl+C is safe: logs are still saved.

## What each run saves

`~/hw3_logs/hw3_<date>_<time>/` on the robot:

| file | what |
|---|---|
| `map_summary.txt` | readable: each tag's x, y, 2-sigma, times seen / used / rejected; final pose |
| `map_final.csv` | final tag positions + covariance (`id,x,y,sxx,sxy,syy`) |
| `map_loops.csv` | the map after every loop |
| `observations.csv` | every detection: robot pose, range/bearing, where that one sighting puts the tag (`seen_x/y`), EKF estimate after it |
| `trajectory.csv` | robot pose + uncertainty after every step |
| `waypoints.csv` | goal vs estimated pose at each waypoint |
| `events.csv` | raw odometry + sightings (input for `replay_slam.py`) |
| `params.json` | all settings used |

The map is in the **robot's start frame**: origin = start position,
+x = starting heading, +y = to its left. The same table prints in the
terminal after every loop and at the end.

## After a run (on the laptop)

Copy the run off the robot (logs are NOT in git):
```bash
scp -r rubikpi:~/hw3_logs/hw3_<date>_<time> ~/cse276a/rubikpi_ros2/hw3_logs/
```
If `rubikpi` doesn't connect, the robot's IP changed: run `hostname -I` on it
and update `HostName` in `~/.ssh/config`.

Then, from `robot_control/robot_control/`:
```bash
# robot's own map, no ground truth needed -> slam_map.png
python3 analyze_map.py LOG_DIR

# compare with ground truth -> map_compare.png, map_errors.csv, error_per_loop.png
python3 analyze_map.py LOG_DIR --gt ground_truth.csv --start X0 Y0 THETA0_DEG \
    --units ft --room 8

# ground-truth sketch for part 1 -> ground_truth_map.png
python3 analyze_map.py --gt ground_truth.csv --start X0 Y0 THETA0_DEG --units ft \
    --room 8 --sketch

# re-run the filter with different noise settings (no robot needed)
python3 replay_slam.py LOG_DIR --k_theta 0.15 --sigma_b_deg 3 --out replay_a
```

## For the report

- **Mean landmark error** (the number the assignment asks for): printed by
  `analyze_map.py --gt`, also in `map_errors.csv`.
- **Error after best rigid fit**: also printed. If it's much smaller than
  the mean error, most of the error came from measuring the start pose by
  hand, not from the map's shape.
- **Square vs octagon**: compare `map_compare.png` and the mean errors. The
  octagon views tags from more angles (expect tighter ellipses); the square
  has fewer, sharper turns.
- **More loops**: `error_per_loop.png` / "After each loop" lines.
- **Simulation (8 ft area, 2 tags per side)**, for reference: mean map error
  2.3 cm (square) and 3.1 cm (octagon) from the center; 7.0 / 8.3 cm from an
  off-center start. Odometry alone drifted 15-53 cm.

## Checks without the robot
```bash
cd robot_control && python3 test/test_ekf_slam.py
```

## Open items
- The 4-tag test run from Nov 6 (`hw3_20261106_003733`) is only on the robot
  in `~/hw3_logs/`. Copy it off with the `scp` above if you want it.
- The robot node (`hw3_slam.py`) hasn't been run since the centered-path and
  logging changes; the filter, replay and plots were tested in simulation.
