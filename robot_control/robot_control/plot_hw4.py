#!/usr/bin/env python3
"""
CSE 276A HW4: figures for the report.

    python3 plot_hw4.py [--config ../config/hw4_params.yaml] [--out hw4_figures]
        [--traj-safe safe_run.csv] [--traj-fast fast_run.csv]

Writes:
    safe_voronoi.png      clearance map, Voronoi diagram, safe path
    fast_visibility.png   configuration space, visibility graph, shortest path
    compare.png           both paths on one map + a metrics table
    compare_runs.png      planned vs driven paths (only with --traj-*)

The trajectory CSVs need columns named x and y (the navigator logs have them).
"""
import argparse
import csv
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle               # noqa: E402

from robot_control.hw4_planning import (DEFAULT_CONFIG, World, load_params,  # noqa: E402
                                        plan_fast, plan_safe)

# Validated categorical slots 1-2 (blue, orange) and neutral inks.
SAFE = '#2a78d6'
FAST = '#eb6834'
SURFACE = '#fcfcfb'
INK = '#1f1f1e'
INK_2 = '#5f5e5a'
INK_3 = '#a3a29c'
OBST = '#3d3d3a'
CLEAR_CMAP = LinearSegmentedColormap.from_list('clear', ['#d9d8d2', '#fcfcfb'])

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': INK_3, 'axes.labelcolor': INK_2, 'xtick.color': INK_2,
    'ytick.color': INK_2, 'text.color': INK, 'font.size': 9, 'axes.titlesize': 11,
    'axes.titleweight': 'bold', 'legend.frameon': False,
})


def base_map(ax, world, title):
    ax.add_patch(Rectangle((0, 0), world.W, world.H, fill=False, ec=INK, lw=1.5, zorder=4))
    for x0, y0, x1, y1 in world.obstacles:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=OBST, ec='none', zorder=4))
    draw_tags(ax, world)
    sx, sy = world.start[:2]
    gx, gy = world.goal
    ax.plot(sx, sy, 'o', ms=9, mfc=SURFACE, mec=INK, mew=2, zorder=7)
    ax.plot(gx, gy, '*', ms=14, mfc=INK, mec=SURFACE, mew=1, zorder=7)
    ax.annotate('Start', (sx, sy), xytext=(8, -12), textcoords='offset points', color=INK, zorder=8)
    ax.annotate('Goal', (gx, gy), xytext=(8, 6), textcoords='offset points', color=INK, zorder=8)
    pad = 0.12
    ax.set_xlim(-pad, world.W + pad)
    ax.set_ylim(-pad, world.H + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title, loc='left')
    for s in ax.spines.values():
        s.set_visible(False)


def draw_tags(ax, world, half=0.08):
    """Each tag: a short bar along its face, a tick showing where it faces, its ID."""
    for tid, (x, y, a) in world.tags.items():
        px, py = -math.sin(a) * half, math.cos(a) * half      # along the face
        ax.plot([x - px, x + px], [y - py, y + py], color=INK, lw=3, solid_capstyle='butt', zorder=6)
        ax.plot([x, x + 0.06 * math.cos(a)], [y, y + 0.06 * math.sin(a)], color=INK, lw=1, zorder=6)
        ax.annotate(str(tid), (x + 0.15 * math.cos(a), y + 0.15 * math.sin(a)),
                    ha='center', va='center', fontsize=7, color=INK_2, zorder=6)


def draw_path(ax, pts, color, label, lw=2, ls='-', markers=True, z=5):
    ax.plot(pts[:, 0], pts[:, 1], ls, color=color, lw=lw, label=label, zorder=z,
            solid_capstyle='round')
    if markers:
        ax.plot(pts[1:-1, 0], pts[1:-1, 1], 'o', ms=8, mfc=color, mec=SURFACE, mew=2, zorder=z + 0.5)


def fig_safe(world, pl, out):
    g = pl.extras['grid']
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    im = ax.imshow(g.d1, origin='lower', extent=(0, world.W, 0, world.H), cmap=CLEAR_CMAP,
                   interpolation='nearest', zorder=1)
    ax.contour(g.xs, g.ys, g.d1, levels=[world.robot_radius, world.r_inf],
               colors=[INK_2, INK_3], linewidths=1, linestyles=['-', '--'], zorder=2)
    ax.plot([], [], color=INK_2, lw=1, label=f'clearance = robot radius ({world.robot_radius:.2f} m)')
    ax.plot([], [], color=INK_3, lw=1, ls='--', label=f'robot radius + margin ({world.r_inf:.2f} m)')
    iy, ix = pl.extras['gvd'].nonzero()
    ax.plot(g.xs[ix], g.ys[iy], 's', ms=1.2, color=INK_2, zorder=3, label='Voronoi diagram')
    for alt in pl.alternatives:
        draw_path(ax, alt, SAFE, 'other side (fewer tags)', lw=1.5, ls='--', markers=False, z=4)
    ax.plot(pl.raw[:, 0], pl.raw[:, 1], color=SAFE, lw=1, alpha=0.5, zorder=4, label='grid path')
    draw_path(ax, pl.points, SAFE, 'safe path (simplified)')
    base_map(ax, world, 'Maximum safety: Voronoi roadmap + Dijkstra')
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('clearance to nearest obstacle (m)')
    cb.outline.set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'safe_voronoi.png'), dpi=200)
    plt.close(fig)


def fig_fast(world, pl, out):
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    bx0, by0, bx1, by1 = pl.extras['bounds']
    ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, ec=INK_3, lw=1,
                           ls='--', zorder=2, label='C-space boundary'))
    for i, (x0, y0, x1, y1) in enumerate(pl.extras['cspace']):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc='#e4e3dd', ec=INK_3, lw=1,
                               hatch='////', zorder=2, label='C-space obstacle' if i == 0 else None))
    nodes = pl.extras['nodes']
    for k, (i, j) in enumerate(pl.extras['edges']):
        ax.plot([nodes[i][0], nodes[j][0]], [nodes[i][1], nodes[j][1]], color=INK_3, lw=0.8,
                zorder=3, label='visibility graph' if k == 0 else None)
    ax.plot([n[0] for n in nodes[2:]], [n[1] for n in nodes[2:]], 'o', ms=5, color=INK_2, zorder=3)
    for alt in pl.alternatives:
        draw_path(ax, alt, FAST, 'other side (fewer tags)', lw=1.5, ls='--', markers=False, z=4)
    draw_path(ax, pl.points, FAST, 'shortest path')
    base_map(ax, world, 'Minimum distance: visibility graph + A*')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fast_visibility.png'), dpi=200)
    plt.close(fig)


ROWS = [('Length (m)', 'length_m', '{:.2f}'),
        ('Waypoints', 'n_waypoints', '{:d}'),
        ('Min clearance, robot edge (m)', 'min_body_clearance_m', '{:.2f}'),
        ('Mean clearance, center (m)', 'mean_clearance_m', '{:.2f}'),
        ('Total turning (deg)', 'total_turn_deg', '{:.0f}'),
        ('Tags visible (mean)', 'mean_tags_visible', '{:.1f}'),
        ('Est. time (s)', 'est_time_s', '{:.0f}')]


def fig_compare(world, safe, fast, out, trajs=None):
    fig, (ax, tx) = plt.subplots(1, 2, figsize=(10, 5.4), gridspec_kw={'width_ratios': [1.25, 1]})
    for pl, color, name in ((safe, SAFE, 'safe'), (fast, FAST, 'fast')):
        draw_path(ax, pl.points, color, f'{name} (planned)', ls='--' if trajs else '-')
        t = (trajs or {}).get(name)
        if t:
            ax.plot(t[0], t[1], color=color, lw=2, label=f'{name} (driven)', zorder=6)
    base_map(ax, world, 'Planned vs driven paths' if trajs else 'Safe vs fast path')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    tx.axis('off')
    cells = [[label, fmt.format(safe.metrics[k]), fmt.format(fast.metrics[k])]
             for label, k, fmt in ROWS]
    tab = tx.table(cellText=cells, colLabels=['', 'Safe', 'Fast'], loc='center',
                   colWidths=[0.62, 0.19, 0.19], cellLoc='right')
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.6)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(INK_3 if r == 0 else '#e4e3dd')
        cell.set_linewidth(0.6)
        cell.set_facecolor(SURFACE)
        if c == 0:
            cell.set_text_props(ha='left', color=INK_2)
            cell._loc = 'left'
        if r == 0:
            cell.set_text_props(weight='bold', color=INK)
    tx.set_title('Planned-path metrics', loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'compare_runs.png' if trajs else 'compare.png'), dpi=200)
    plt.close(fig)


def read_traj(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return [float(r['x']) for r in rows], [float(r['y']) for r in rows]


def main():
    ap = argparse.ArgumentParser(description='HW4 report figures')
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--out', default='hw4_figures')
    ap.add_argument('--traj-safe')
    ap.add_argument('--traj-fast')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    world = World(load_params(a.config))
    safe, fast = plan_safe(world), plan_fast(world)
    fig_safe(world, safe, a.out)
    fig_fast(world, fast, a.out)
    fig_compare(world, safe, fast, a.out)
    trajs = {k: read_traj(p) for k, p in (('safe', a.traj_safe), ('fast', a.traj_fast)) if p}
    if trajs:
        fig_compare(world, safe, fast, a.out, trajs)
    print(f'figures written to {os.path.abspath(a.out)}')


if __name__ == '__main__':
    main()
