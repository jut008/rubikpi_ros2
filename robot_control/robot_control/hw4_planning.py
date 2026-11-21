#!/usr/bin/env python3
"""
CSE 276A HW4: two path planners for the 10 ft x 10 ft arena.

safe : maximum clearance.
       World representation: an occupancy grid with an exact clearance map
       (distance from every cell to the nearest obstacle, treating the arena
       boundary as one obstacle and each box as another).
       Roadmap: the generalized Voronoi diagram (GVD), the cells that are
       equally far from the two nearest obstacles. Every point on it is as
       far from obstacles as it can be.
       Search: the start and goal are "retracted" onto the GVD by climbing
       the clearance map, then Dijkstra finds the shortest route along it.
       The grid path is then simplified to a few waypoints without giving
       up more than `clearance_slack` of clearance.

fast : minimum distance.
       World representation: configuration space. Each obstacle is grown by
       robot_radius + safety_margin and the arena is shrunk by the same
       amount, so the robot can be treated as a point.
       Roadmap: the visibility graph. Nodes are the start, the goal and the
       corners of the grown obstacles; edges join every pair that can see
       each other. With polygon obstacles the shortest path is always on
       this graph.
       Search: A* with straight-line distance as the heuristic.

When two routes are (almost) the same length, for example going around
either side of a centered obstacle, the one where the camera can see more
AprilTags wins.

No ROS in here. The navigator imports it, and it runs on a laptop:
    python3 hw4_planning.py [--config ../config/hw4_params.yaml]
"""
import argparse
import heapq
import math
import os
from dataclasses import dataclass, field

import numpy as np
import yaml

DEFAULT_CONFIG = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'hw4_params.yaml'))


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def load_params(path=DEFAULT_CONFIG):
    """ros__parameters dict from a ROS params YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return next(iter(data.values()))['ros__parameters']


# ---------------------------------------------------------------- geometry

def rect_distance(x, y, rect):
    """Distance from points (x, y) to an axis-aligned box (0 inside)."""
    x0, y0, x1, y1 = rect
    dx = np.maximum(np.maximum(x0 - x, x - x1), 0.0)
    dy = np.maximum(np.maximum(y0 - y, y - y1), 0.0)
    return np.hypot(dx, dy)


def inflate(rect, r):
    x0, y0, x1, y1 = rect
    return (x0 - r, y0 - r, x1 + r, y1 + r)


def seg_hits_rect(a, b, rect, eps=1e-6):
    """True if segment a-b passes through the interior of the box
    (touching an edge or a corner does not count). Liang-Barsky clipping."""
    x0, y0, x1, y1 = rect[0] + eps, rect[1] + eps, rect[2] - eps, rect[3] - eps
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return t1 - t0 > 1e-9


def point_segment_distance(p, a, b):
    """Distance from points p (N x 2) to segment a-b."""
    p, a, b = np.asarray(p, float), np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return np.linalg.norm(p - a, axis=1)
    t = np.clip((p - a) @ ab / L2, 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def path_length(pts):
    pts = np.asarray(pts, float)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


# ---------------------------------------------------------------- world

class World:
    def __init__(self, p):
        self.p = p
        self.W, self.H = (float(v) for v in p['arena_size'])
        o = [float(v) for v in p['obstacles']]
        self.obstacles = [tuple(o[i:i + 4]) for i in range(0, len(o), 4)]
        self.start = tuple(float(v) for v in p['start'])
        g = [float(v) for v in p['goal']]
        self.goal = (g[0], g[1])
        self.goal_theta = g[2] if len(g) > 2 else None
        self.robot_radius = float(p['robot_radius'])
        self.r_inf = self.robot_radius + float(p['safety_margin'])
        self.tags = {int(i): (float(x), float(y), float(a)) for i, x, y, a in
                     zip(p['tag_ids'], p['tag_x'], p['tag_y'], p['tag_yaw'])}
        self.cam_offset = (float(p.get('camera_offset_x', 0.0)),
                           float(p.get('camera_offset_y', 0.0)))

    def site_distances(self, x, y):
        """Stack of distances to each obstacle: [boundary, box 1, box 2, ...]."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        walls = np.minimum(np.minimum(x, self.W - x), np.minimum(y, self.H - y))
        out = [np.maximum(walls, 0.0)]
        out += [rect_distance(x, y, r) for r in self.obstacles]
        return np.stack(out)

    def clearance(self, x, y):
        """Distance from the robot center to the nearest obstacle or edge."""
        return self.site_distances(x, y).min(axis=0)

    def segment_clearance(self, a, b, step=0.01):
        n = max(2, int(math.ceil(math.dist(a, b) / step)) + 1)
        t = np.linspace(0.0, 1.0, n)
        return float(self.clearance(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])).min())

    def line_of_sight(self, a, b):
        return not any(seg_hits_rect(a, b, r) for r in self.obstacles)

    def visible_tags(self, x, y, th):
        """IDs of tags the camera should be able to read from pose (x, y, th)."""
        p = self.p
        hfov = float(p.get('camera_hfov', 1.0))
        max_d = float(p.get('max_tag_dist', 2.5))
        max_view = float(p.get('max_tag_view_angle', 1.05))
        ox, oy = self.cam_offset
        cx = x + ox * math.cos(th) - oy * math.sin(th)
        cy = y + ox * math.sin(th) + oy * math.cos(th)
        seen = []
        for tid, (tx, ty, ta) in self.tags.items():
            dx, dy = tx - cx, ty - cy
            d = math.hypot(dx, dy)
            if d < 1e-6 or d > max_d:
                continue
            if abs(wrap(math.atan2(dy, dx) - th)) > hfov / 2:      # outside the FOV
                continue
            if abs(wrap(math.atan2(-dy, -dx) - ta)) > max_view:    # seen too edge-on / from behind
                continue
            if not self.line_of_sight((cx, cy), (tx, ty)):         # hidden by the obstacle
                continue
            seen.append(tid)
        return seen


# ---------------------------------------------------------------- results

@dataclass
class Plan:
    name: str
    points: np.ndarray                 # simplified path, N x 2 (start ... goal)
    waypoints: list                    # [(x, y, theta), ...] for the navigator
    metrics: dict
    raw: np.ndarray = None             # unsimplified path (safe planner)
    alternatives: list = field(default_factory=list)   # other candidate paths
    extras: dict = field(default_factory=dict)         # data for plots


def to_waypoints(world, pts):
    """(x, y, theta) list. Each waypoint's theta is the heading of the next
    leg, so the robot turns at the waypoint and then drives straight."""
    pts = np.asarray(pts, float)
    heads = [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(pts[:-1], pts[1:])]
    wps = [(float(pts[0][0]), float(pts[0][1]), float(world.start[2]))]
    for k in range(1, len(pts)):
        th = heads[k] if k < len(heads) else (
            world.goal_theta if world.goal_theta is not None else heads[-1])
        wps.append((float(pts[k][0]), float(pts[k][1]), float(wrap(th))))
    return wps


def sample_path(pts, step=0.1):
    """Poses (x, y, heading) every `step` meters along the path."""
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        L = math.dist(a, b)
        h = math.atan2(b[1] - a[1], b[0] - a[0])
        for t in np.arange(0.0, L, step):
            out.append((a[0] + (b[0] - a[0]) * t / L, a[1] + (b[1] - a[1]) * t / L, h))
    out.append((pts[-1][0], pts[-1][1], out[-1][2] if out else 0.0))
    return out


def path_metrics(world, pts):
    p = world.p
    pts = np.asarray(pts, float)
    legs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    heads = [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(pts[:-1], pts[1:])]
    turns = [abs(wrap(heads[0] - world.start[2]))]
    turns += [abs(wrap(h2 - h1)) for h1, h2 in zip(heads[:-1], heads[1:])]
    if world.goal_theta is not None:
        turns.append(abs(wrap(world.goal_theta - heads[-1])))
    n_turns = sum(t > 0.05 for t in turns)

    clear = min(world.segment_clearance(a, b) for a, b in zip(pts[:-1], pts[1:]))
    samples = sample_path(pts)
    sample_clear = world.clearance([s[0] for s in samples], [s[1] for s in samples])
    seen = [len(world.visible_tags(*s)) for s in samples]

    v = float(p.get('drive_speed', 0.5))
    step = v * float(p.get('step_time', 1.0))
    n_steps = int(sum(math.ceil(L / step) for L in legs)) + n_turns
    drive_t = float(legs.sum()) / v
    turn_t = sum(turns) / float(p.get('turn_rate', 2.0))
    est_t = drive_t + turn_t + n_steps * float(p.get('step_overhead', 1.0))

    return {
        'length_m': float(legs.sum()),
        'n_waypoints': len(pts),
        'min_clearance_m': clear,                         # robot center to obstacle
        'min_body_clearance_m': clear - world.robot_radius,  # robot edge to obstacle
        'mean_clearance_m': float(sample_clear.mean()),
        'total_turn_deg': math.degrees(sum(turns)),
        'n_turns': n_turns,
        'mean_tags_visible': float(np.mean(seen)),
        'frac_no_tag': float(np.mean([s == 0 for s in seen])),
        'est_time_s': est_t,
    }


def pick(world, candidates, tie_tol):
    """Shortest candidate, except that among near-ties (within tie_tol) the
    one that sees the most tags wins."""
    lens = [path_length(c) for c in candidates]
    best = min(lens)
    ties = [c for c, L in zip(candidates, lens) if L <= best * (1.0 + tie_tol)]
    if len(ties) == 1:
        return ties[0]
    return max(ties, key=lambda c: path_metrics(world, c)['mean_tags_visible'])


# ---------------------------------------------------------------- safe: Voronoi

NBRS = [(-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)), (0, -1, 1.0),
        (0, 1, 1.0), (1, -1, math.sqrt(2)), (1, 0, 1.0), (1, 1, math.sqrt(2))]


class Grid:
    """Cell (iy, ix) covers x in [ix*res, (ix+1)*res), same for y."""

    def __init__(self, world, res):
        self.res = res
        self.nx = int(round(world.W / res))
        self.ny = int(round(world.H / res))
        self.xs = (np.arange(self.nx) + 0.5) * res
        self.ys = (np.arange(self.ny) + 0.5) * res
        X, Y = np.meshgrid(self.xs, self.ys)
        D = world.site_distances(X, Y)
        Ds = np.sort(D, axis=0)
        self.d1 = Ds[0]                 # clearance
        self.d2 = Ds[1]                 # distance to the second-nearest obstacle
        self.nearest = D.argmin(axis=0)

    def cell(self, x, y):
        return (min(max(int(y / self.res), 0), self.ny - 1),
                min(max(int(x / self.res), 0), self.nx - 1))

    def xy(self, c):
        return ((c[1] + 0.5) * self.res, (c[0] + 0.5) * self.res)

    def inside(self, c):
        return 0 <= c[0] < self.ny and 0 <= c[1] < self.nx

    def neighbors(self, c):
        for di, dj, w in NBRS:
            n = (c[0] + di, c[1] + dj)
            if self.inside(n):
                yield n, w


def retract(grid, gvd, free, c):
    """Climb the clearance map from cell c until reaching the GVD. Falls back
    to a breadth-first search for the nearest GVD cell if the climb stalls."""
    path = [c]
    while not gvd[c]:
        best = max(grid.neighbors(c), key=lambda nw: grid.d1[nw[0]])[0]
        if grid.d1[best] <= grid.d1[c]:
            break
        c = best
        path.append(c)
    if gvd[c]:
        return path
    prev = {c: None}
    queue = [c]
    for cur in queue:
        if gvd[cur]:
            tail = []
            while cur is not None and cur != c:
                tail.append(cur)
                cur = prev[cur]
            return path + tail[::-1]
        for n, _ in grid.neighbors(cur):
            if n not in prev and free[n]:
                prev[n] = cur
                queue.append(n)
    return None


def grid_dijkstra(grid, mask, s, g, banned=frozenset()):
    dist = {s: 0.0}
    prev = {}
    pq = [(0.0, s)]
    while pq:
        d, c = heapq.heappop(pq)
        if c == g:
            break
        if d > dist[c]:
            continue
        for n, w in grid.neighbors(c):
            if not mask[n] or n in banned:
                continue
            nd = d + w * grid.res
            if nd < dist.get(n, math.inf):
                dist[n] = nd
                prev[n] = c
                heapq.heappush(pq, (nd, n))
    if g not in dist:
        return None
    path = [g]
    while path[-1] != s:
        path.append(prev[path[-1]])
    return path[::-1]


def simplify(world, pts, tol, slack):
    """Douglas-Peucker that also refuses a shortcut if it gets closer to an
    obstacle than the original path did there (minus `slack`)."""
    pts = np.asarray(pts, float)
    n = len(pts)
    clear = world.clearance(pts[:, 0], pts[:, 1])
    keep = {0, n - 1}
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        dev = point_segment_distance(pts[i + 1:j], pts[i], pts[j])
        k = i + 1 + int(np.argmax(dev))
        orig = float(clear[i:j + 1].min())
        # don't get closer than the original path did (minus slack), but never
        # demand more than the original had: near a start or goal that sits
        # inside the safety margin, the original itself is closer than r_inf
        need = max(orig - slack, min(world.r_inf, orig))
        if dev.max() > tol or world.segment_clearance(pts[i], pts[j]) < need:
            keep.add(k)
            stack += [(i, k), (k, j)]
    return pts[sorted(keep)]


def drop_short_legs(world, pts, min_leg):
    """Remove interior points that make a leg shorter than min_leg, when the
    resulting shortcut keeps the clearance (tiny legs mean tiny, unreliable turns)."""
    pts = [tuple(p) for p in pts]
    changed = True
    while changed and len(pts) > 2:
        changed = False
        for k in range(1, len(pts) - 1):
            if min(math.dist(pts[k - 1], pts[k]), math.dist(pts[k], pts[k + 1])) >= min_leg:
                continue
            old = min(world.segment_clearance(pts[k - 1], pts[k]),
                      world.segment_clearance(pts[k], pts[k + 1]))
            if world.segment_clearance(pts[k - 1], pts[k + 1]) >= min(old, world.r_inf):
                del pts[k]
                changed = True
                break
    return np.array(pts)


def plan_safe(world):
    p = world.p
    res = float(p.get('grid_res', 0.02))
    grid = Grid(world, res)
    free = grid.d1 >= world.robot_radius
    s_cell = grid.cell(world.start[0], world.start[1])
    g_cell = grid.cell(*world.goal)
    if not free[s_cell] or not free[g_cell]:
        raise ValueError('start or goal is closer to an obstacle than robot_radius')

    # The GVD band must be a few cells thick to stay connected on a grid.
    for k in (1.0, 1.5, 2.0, 3.0):
        gvd = (grid.d2 - grid.d1 <= k * res) & (grid.d1 >= world.r_inf)
        rs = retract(grid, gvd, free, s_cell)
        rg = retract(grid, gvd, free, g_cell)
        if rs is None or rg is None:
            continue
        core = grid_dijkstra(grid, gvd, rs[-1], rg[-1])
        if core is not None:
            break
    else:
        raise RuntimeError('no path along the Voronoi diagram')

    cores = [core]
    # The other way around: block the middle of the first route and search again.
    cut = int(round(0.3 / res))
    if len(core) > 2 * cut + 2:
        banned = set()
        for c in core[cut:-cut]:
            for di in range(-3, 4):
                for dj in range(-3, 4):
                    banned.add((c[0] + di, c[1] + dj))
        alt = grid_dijkstra(grid, gvd, rs[-1], rg[-1], banned)
        if alt is not None:
            cores.append(alt)

    tol = float(p.get('simplify_tol', 0.05))
    slack = float(p.get('clearance_slack', 0.03))
    raws, simps = [], []
    for c in cores:
        cells = rs[:-1] + c + rg[::-1][1:]
        raw = np.array([world.start[:2]] + [grid.xy(x) for x in cells[1:-1]] + [world.goal])
        raws.append(raw)
        simps.append(drop_short_legs(world, simplify(world, raw, tol, slack),
                                     float(p.get('min_leg', 0.15))))

    best = pick(world, simps, float(p.get('tie_tol', 0.02)))
    i = next(k for k, s in enumerate(simps) if s is best)
    return Plan('safe', best, to_waypoints(world, best), path_metrics(world, best),
                raw=raws[i],
                alternatives=[s for s in simps if s is not best],
                extras={'grid': grid, 'gvd': gvd})


# ---------------------------------------------------------------- fast: visibility graph

def astar(nodes, adj, s, g, banned=frozenset()):
    h = lambda i: math.dist(nodes[i], nodes[g])
    dist = {s: 0.0}
    prev = {}
    pq = [(h(s), s)]
    done = set()
    while pq:
        _, i = heapq.heappop(pq)
        if i == g:
            break
        if i in done:
            continue
        done.add(i)
        for j, w in adj[i]:
            if j in banned:
                continue
            nd = dist[i] + w
            if nd < dist.get(j, math.inf):
                dist[j] = nd
                prev[j] = i
                heapq.heappush(pq, (nd + h(j), j))
    if g not in dist:
        return None
    path = [g]
    while path[-1] != s:
        path.append(prev[path[-1]])
    return path[::-1]


def plan_fast(world):
    p = world.p
    r = world.r_inf
    cspace = [inflate(o, r) for o in world.obstacles]
    bounds = (r, r, world.W - r, world.H - r)       # where the robot center may go

    def in_free(q, b=bounds):
        return (b[0] <= q[0] <= b[2] and b[1] <= q[1] <= b[3]
                and not any(c[0] < q[0] < c[2] and c[1] < q[1] < c[3] for c in cspace))

    nodes = [tuple(world.start[:2]), tuple(world.goal)]
    # The robot starts and ends where it is put: the start and goal only need to
    # clear the walls by robot_radius (not the margin) and stay out of the boxes.
    rr = world.robot_radius
    if not all(in_free(q, (rr, rr, world.W - rr, world.H - rr)) for q in nodes):
        raise ValueError('start or goal is inside the configuration-space obstacles')
    e = 1e-3     # nudge corners outward so edges along an obstacle side are not "inside" it
    for x0, y0, x1, y1 in cspace:
        for q in ((x0 - e, y0 - e), (x1 + e, y0 - e), (x1 + e, y1 + e), (x0 - e, y1 + e)):
            if in_free(q):
                nodes.append(q)

    adj = {i: [] for i in range(len(nodes))}
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            if not any(seg_hits_rect(a, b, c) for c in cspace):
                w = math.dist(a, b)
                adj[i].append((j, w))
                adj[j].append((i, w))
                edges.append((i, j))

    best = astar(nodes, adj, 0, 1)
    if best is None:
        raise RuntimeError('no path in the visibility graph')
    routes = [best] + [alt for alt in (astar(nodes, adj, 0, 1, {k}) for k in best[1:-1]) if alt]
    paths = [np.array([nodes[i] for i in rt]) for rt in routes]
    chosen = pick(world, paths, float(p.get('tie_tol', 0.02)))
    return Plan('fast', chosen, to_waypoints(world, chosen), path_metrics(world, chosen),
                alternatives=[q for q in paths if q is not chosen],
                extras={'nodes': nodes, 'edges': edges, 'cspace': cspace, 'bounds': bounds})


def plan(world, which):
    return {'safe': plan_safe, 'fast': plan_fast}[which](world)


# ---------------------------------------------------------------- CLI

def report(pl):
    m = pl.metrics
    lines = [f'== {pl.name} ==',
             f'  length           {m["length_m"]:.2f} m',
             f'  waypoints        {m["n_waypoints"]}',
             f'  min clearance    {m["min_clearance_m"]:.3f} m center, '
             f'{m["min_body_clearance_m"]:.3f} m robot edge',
             f'  mean clearance   {m["mean_clearance_m"]:.3f} m',
             f'  turning          {m["total_turn_deg"]:.0f} deg in {m["n_turns"]} turns',
             f'  tags visible     {m["mean_tags_visible"]:.1f} on average, '
             f'{100 * m["frac_no_tag"]:.0f}% of the path sees none',
             f'  est. time        {m["est_time_s"]:.0f} s (stop-and-go)',
             '  waypoints (x, y, theta):']
    lines += [f'    {x:6.3f} {y:6.3f} {math.degrees(t):7.1f} deg' for x, y, t in pl.waypoints]
    flat = ', '.join(f'{v:.4f}' for wp in pl.waypoints for v in wp)
    lines.append(f'  yaml: waypoints: [{flat}]')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--planner', choices=['safe', 'fast', 'both'], default='both')
    a = ap.parse_args()
    world = World(load_params(a.config))
    for name in (['safe', 'fast'] if a.planner == 'both' else [a.planner]):
        print(report(plan(world, name)))


if __name__ == '__main__':
    main()
