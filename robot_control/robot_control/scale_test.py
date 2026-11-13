#!/usr/bin/env python3
"""
Replay a hw3 run with the drive distances scaled, to see how far the robot
really moved per drive. Usage:
    python3 scale_test.py ~/hw3_logs/hw3_20261112_233514
Finds replay_slam.py by itself (under ~/ros2_ws), or pass its path as a 2nd argument.
"""
import csv, glob, importlib.util, os, shutil, sys, tempfile

log = os.path.expanduser(sys.argv[1])
if len(sys.argv) > 2:
    hint = os.path.expanduser(sys.argv[2])
    cands = [hint] if hint.endswith('.py') else glob.glob(os.path.join(hint, '**', 'replay_slam.py'), recursive=True)
else:
    cands = glob.glob(os.path.expanduser('~/ros2_ws/src/**/replay_slam.py'), recursive=True)
cands = [c for c in cands if os.path.isfile(c) and '/install/' not in c and '/build/' not in c]
if not cands:
    sys.exit('Could not find replay_slam.py. Pass its path: python3 scale_test.py LOG_DIR /path/to/replay_slam.py')
path = cands[0]
print(f'Using {path}')
spec = importlib.util.spec_from_file_location('replay_slam', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
if not hasattr(mod, 'replay'):
    sys.exit(f'{path} has no replay() function; is it the hw3 branch version?')
for need in ('events.csv', 'params.json'):
    if not os.path.isfile(os.path.join(log, need)):
        sys.exit(f'{os.path.join(log, need)} not found; LOG_DIR must be a hw3_<date>_<time> run folder')

rows = list(csv.reader(open(os.path.join(log, 'events.csv'))))
print(' drive scale   accepted  rejected')
for k in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
    tmp = tempfile.mkdtemp()
    shutil.copy(os.path.join(log, 'params.json'), tmp)
    with open(os.path.join(tmp, 'events.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(rows[0])
        for r in rows[1:]:
            r = list(r)
            if r[1] == 'predict':
                r[2] = str(float(r[2]) * k)
            w.writerow(r)
    c = mod.replay(tmp)[4]
    print(f'   {k:4.1f}        {c["updated"]:5d}    {c["rejected"]:5d}')
    shutil.rmtree(tmp)
