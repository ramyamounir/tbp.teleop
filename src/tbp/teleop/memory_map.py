# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation, Slerp

from tbp.teleop.helpers import is_interactive_backend

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import TimerBase
    from matplotlib.collections import PathCollection
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    from tbp.monty.frameworks.models.graph_matching import GraphLM

_FADE_COLOR = np.array([0.85, 0.1, 0.1])
_FALLBACK_COLOR = np.array([0.25, 0.25, 0.25])
# The sliver of data space kept below the map so bottom-row labels are not clipped.
_LABEL_MARGIN = 0.04
# The fraction of the figure's subplot-area width the overlay covers, centered.
_WIDTH_FRACTION = 0.75
# One full idle revolution of every cloud takes this long.
_SPIN_PERIOD_SECONDS = 15.0
# How often the idle-spin timer repaints the spinning clouds.
_SPIN_INTERVAL_MS = 66
# How strongly far-side nodes are dimmed to give the spinning clouds depth.
_DEPTH_DIM = 0.6


def _spin_pose(angle: float) -> npt.NDArray[np.float64]:
    """Build the pose for a spin of `angle` radians around the vertical axis.

    Args:
        angle: The spin angle in radians.

    Returns:
        The `(3, 3)` rotation matrix mixing the data x and z axes.
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _nearest_rotation(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Project a matrix onto the nearest proper rotation.

    Hypothesis poses can come out left-handed on merged graphs (feat.janus'
    `pose_vector_mean` fallbacks), and scipy refuses to interpolate those.

    Args:
        matrix: The `(3, 3)` matrix to project.

    Returns:
        The closest `(3, 3)` rotation matrix with determinant +1.
    """
    u, _, vt = np.linalg.svd(matrix)
    flip = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    return u @ np.diag([1.0, 1.0, flip]) @ vt


@dataclass
class _Cloud:
    """One object model's display state on the map.

    The graph's 3D nodes are stored normalized to a unit radius around their
    midrange center, so the cloud can be drawn at any layout position, display
    radius, and orientation by an affine transform alone. `pose` is the cloud's
    current orientation; the idle spin advances it around the vertical axis
    while a merge alignment sets it directly.
    """

    unit_points: npt.NDArray[np.float64]
    colors: npt.NDArray[np.float64]
    num_points: int
    center: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(2, dtype=float)
    )
    pose: npt.NDArray[np.float64] = field(default_factory=lambda: np.eye(3))
    angle: float = 0.0
    scatter: PathCollection | None = None
    label: Text | None = None

    def spin(self, delta: float) -> None:
        """Advance the idle rotation around the vertical axis.

        Args:
            delta: The angle increment in radians.
        """
        self.angle = (self.angle + delta) % math.tau
        self.pose = _spin_pose(self.angle)

    def place(self, radius: float, alpha: float = 1.0, shift: float = 0.0) -> None:
        """Reposition the cloud's artists at its current center and pose.

        Nodes are drawn back-to-front with far-side nodes dimmed, so the
        rotating cloud reads as a solid 3D object.

        Args:
            radius: The display radius in data coordinates.
            alpha: The opacity of both the nodes and the label.
            shift: Fraction in [0, 1] by which the node colors are blended
                toward the fade-out red.
        """
        rotated = self.unit_points @ self.pose.T
        order = np.argsort(rotated[:, 2])
        rotated = rotated[order]
        colors = self.colors[order] * (1.0 - shift) + _FADE_COLOR * shift
        depth = np.clip((rotated[:, 2] + 1.0) / 2.0, 0.0, 1.0)
        shade = _DEPTH_DIM + (1.0 - _DEPTH_DIM) * depth
        rgba = np.concatenate(
            [colors * shade[:, None], np.full((len(colors), 1), alpha)],
            axis=1,
        )
        self.scatter.set_offsets(self.center + rotated[:, :2] * radius)
        self.scatter.set_facecolor(rgba)
        self.label.set_position((self.center[0], self.center[1] - radius - 0.012))
        self.label.set_alpha(alpha)
        self.label.set_color(_FADE_COLOR if shift > 0 else "0.25")

    def remove(self) -> None:
        """Remove the cloud's artists from the axis."""
        self.scatter.remove()
        self.label.remove()


class MemoryMap:
    """A live 2D map of every object model in a learning module's memory.

    Drawn as a toggleable overlay on the teleop figure itself: an opaque axes
    covering the panel area (roughly 70% of the figure surface) that sits above
    the panels while open and is removed when toggled off. Each object graph is
    drawn as its hsv-colored node cloud, labeled `model_name (pts)`, at a
    jittered non-overlapping grid position filling the 2D space. All clouds spin
    slowly and continuously around the vertical axis (driven by a canvas timer,
    so they keep turning while the teleop waits for input).

    On every `update` the current graph memory ids are diffed against the previous
    step's. When ids disappear and a new id appears in the same step (a merge under
    `EvidenceGraphLM.update_terminal_condition`), the source clouds first rotate
    into the merged object's reference frame using their last most-likely-hypothesis
    poses, then turn red, fade out, and glide toward the merged object's spot while
    the merged cloud fades in upright; the surviving clouds then own larger grid
    cells and glide to re-fill the freed space. Plain additions fade in and plain
    removals fade out red, both followed by the same smooth relayout.
    """

    def __init__(
        self,
        max_display_points: int = 300,
        animation_frames: int = 24,
        frame_delay: float = 0.02,
    ) -> None:
        """Initialize the memory map.

        Args:
            max_display_points: Maximum graph nodes drawn per object; larger graphs
                are randomly downsampled for display (the label keeps the true
                count).
            animation_frames: Number of frames per merge / relayout animation.
            frame_delay: Pause in seconds between animation frames.
        """
        self.max_display_points = max_display_points
        self.animation_frames = animation_frames
        self.frame_delay = frame_delay

        self._fig: Figure | None = None
        self._ax: Axes | None = None
        self._lm: GraphLM | None = None
        self._clouds: dict[str, _Cloud] = {}
        self._radius = 0.1
        self._rng = np.random.default_rng(42)
        self._timer: TimerBase | None = None
        # Blocks the idle-spin timer while a scripted animation drives the frames.
        self._animating = False
        # Last known MLH rotation per graph id; the merge drops the source
        # hypotheses before `update` sees the merged id, so these are cached
        # every step to drive the pre-merge alignment.
        self._mlh_poses: dict[str, npt.NDArray[np.float64]] = {}
        # Data space: unit height, width matching the overlay's aspect ratio;
        # computed from the host figure when the overlay axes is built.
        self._extent = (1.0, 1.0)

    @property
    def lm(self) -> GraphLM | None:
        """The learning module currently mapped, or `None` before `initialize`."""
        return self._lm

    def initialize(self, fig: Figure, lm: GraphLM) -> None:
        """Bind to a learning module and draw its memory as an overlay on `fig`.

        Creates the overlay axes on first use (or when the host figure was
        rebuilt); rebinding to a different learning module redraws the map with
        that module's memory.

        Args:
            fig: The teleop figure the overlay is drawn on.
            lm: The learning module whose graph memory is mapped.
        """
        if self._ax is None or self._ax.figure is not fig:
            # Any previous artists died with the old figure.
            self._clouds = {}
            self._build_axes(fig)
        self._fig = fig
        self._lm = lm
        for cloud in self._clouds.values():
            cloud.remove()
        self._clouds = {}
        self._mlh_poses = {}
        snapshot = self._snapshot()
        centers, self._radius = self._layout(list(snapshot), {})
        for graph_id, cloud in snapshot.items():
            cloud.center = centers[graph_id]
            # Random spin phases so the clouds don't all turn in lockstep.
            cloud.spin(self._rng.uniform(0.0, math.tau))
            self._add_artists(graph_id, cloud)
            cloud.place(self._radius)
            self._clouds[graph_id] = cloud
        self._set_title()
        self._draw_frame()
        self._start_timer()

    def update(self) -> None:
        """Diff the graph memory against the last drawn state and animate changes.

        Cheap when nothing changed: only the memory ids and node counts are read.
        """
        if self._fig is None or self._lm is None:
            return
        current_ids = list(self._lm.graph_memory.get_memory_ids())
        self._cache_mlh_poses(current_ids)
        removed = [g for g in self._clouds if g not in current_ids]
        added = [g for g in current_ids if g not in self._clouds]
        if not removed and not added:
            self._refresh_grown_clouds()
            return

        snapshot = self._snapshot()
        target = added[0] if removed and added else None
        self._animating = True
        try:
            if target is not None:
                self._align_for_merge(removed, target)
            self._animate_transition(snapshot, added, removed, target)
        finally:
            self._animating = False
        self._mlh_poses = {
            g: p for g, p in self._mlh_poses.items() if g in self._clouds
        }

    def close(self) -> None:
        """Remove the overlay from its host figure and drop its artists."""
        self._stop_timer()
        if self._ax is not None:
            self._ax.remove()
        if self._fig is not None:
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
        self._fig = None
        self._ax = None
        self._clouds = {}

    def _build_axes(self, fig: Figure) -> None:
        """Create the overlay axes over the host figure's subplot area.

        The overlay spans the subplot area's full height but only
        `_WIDTH_FRACTION` of its width, centered.

        Args:
            fig: The teleop figure the overlay is drawn on.
        """
        margins = fig.subplotpars
        width = (margins.right - margins.left) * _WIDTH_FRACTION
        center = (margins.left + margins.right) / 2
        rect = (
            center - width / 2,
            margins.bottom,
            width,
            margins.top - margins.bottom,
        )
        width_in, height_in = fig.get_size_inches()
        box_aspect = (width_in * rect[2]) / (height_in * rect[3])
        self._extent = (box_aspect * (1.0 + _LABEL_MARGIN), 1.0)

        self._ax = fig.add_axes(rect)
        # Draw above the panels and the figure-level section dividers, with an
        # opaque background so the overlay fully covers them. The high zorder
        # also makes covered widgets ignore clicks while the overlay is open.
        self._ax.set_zorder(10)
        self._ax.set_facecolor("white")
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._ax.set_xlim(0.0, self._extent[0])
        self._ax.set_ylim(-_LABEL_MARGIN, self._extent[1])
        self._ax.set_aspect("equal")

    def _set_title(self) -> None:
        """Show the mapped learning module and its current object count.

        The title is pulled inside the overlay's top edge so it doesn't collide
        with the selector buttons in the figure margin above.
        """
        self._ax.set_title(
            f"{self._lm.learning_module_id} memory · {len(self._clouds)} objects",
            fontsize=11,
            y=1.0,
            pad=-16,
        )

    def _snapshot(self) -> dict[str, _Cloud]:
        """Read every object model in the LM's graph memory into display clouds.

        Each graph's first input channel supplies the node locations and the `hsv`
        feature for the node colors (dark gray when the feature is absent).

        Returns:
            A mapping from graph id to its normalized display cloud.
        """
        memory = self._lm.graph_memory
        clouds: dict[str, _Cloud] = {}
        for graph_id in memory.get_memory_ids():
            channels = memory.get_input_channels_in_graph(graph_id)
            if not channels:
                continue
            channel = channels[0]
            pos = np.asarray(memory.get_locations_in_graph(graph_id, channel))
            if pos.ndim != 2 or pos.shape[0] == 0:
                continue
            hsv = memory.get_features_by_name(graph_id, channel).get("hsv")
            clouds[graph_id] = self._make_cloud(pos, hsv)
        return clouds

    def _make_cloud(
        self,
        pos: npt.NDArray[np.float64],
        hsv: npt.NDArray[np.float64] | None,
    ) -> _Cloud:
        """Build one display cloud from a graph's node locations and hsv feature.

        Args:
            pos: The `(N, 3)` node locations of the graph.
            hsv: The `(N, >=3)` hsv feature aligned with `pos`, or `None`.

        Returns:
            The cloud with its 3D nodes normalized to a unit radius, so any
            rotation of the cloud stays inside its display circle.
        """
        num_points = pos.shape[0]
        if hsv is not None:
            hsv = np.asarray(hsv, dtype=float)
            if hsv.ndim != 2 or hsv.shape[0] != num_points or hsv.shape[1] < 3:
                hsv = None

        if num_points > self.max_display_points:
            keep = self._rng.choice(
                num_points, size=self.max_display_points, replace=False
            )
            pos = pos[keep]
            hsv = hsv[keep] if hsv is not None else None

        pts = pos.astype(float)
        low, high = pts.min(axis=0), pts.max(axis=0)
        pts = pts - (low + high) / 2
        norm = float(np.linalg.norm(pts, axis=1).max())
        if norm > 1e-12:
            pts = pts / norm

        if hsv is not None:
            colors = mcolors.hsv_to_rgb(np.clip(hsv[:, :3], 0.0, 1.0))
        else:
            colors = np.tile(_FALLBACK_COLOR, (len(pts), 1))
        return _Cloud(unit_points=pts, colors=colors, num_points=num_points)

    def _layout(
        self,
        graph_ids: list[str],
        previous_centers: dict[str, npt.NDArray[np.float64]],
    ) -> tuple[dict[str, npt.NDArray[np.float64]], float]:
        """Assign every object a jittered, non-overlapping grid cell.

        The grid is sized to the object count, so fewer objects yield larger cells
        (and clouds) that fill the same space. Objects with a previous position take
        the free cell nearest to it, keeping the relayout glide short; objects
        without one (e.g. a freshly merged graph) land in a random free cell.

        Args:
            graph_ids: The graph ids to place.
            previous_centers: The last known center per graph id, for stability.

        Returns:
            The center per graph id and the shared display radius.
        """
        width, height = self._extent
        n = max(len(graph_ids), 1)
        cols = max(1, round(math.sqrt(n * width / height)))
        rows = math.ceil(n / cols)
        while cols * rows < n:
            cols += 1
        cell = (width / cols, height / rows)
        radius = 0.36 * min(cell)

        cells = [
            np.array([(c + 0.5) * cell[0], (r + 0.5) * cell[1]])
            for r in range(rows)
            for c in range(cols)
        ]
        free = list(range(len(cells)))
        centers: dict[str, npt.NDArray[np.float64]] = {}

        known = [g for g in graph_ids if g in previous_centers]
        new = [g for g in graph_ids if g not in previous_centers]
        for graph_id in known:
            old = previous_centers[graph_id]
            best = min(free, key=lambda i: float(np.sum((cells[i] - old) ** 2)))
            free.remove(best)
            centers[graph_id] = cells[best]
        for graph_id in new:
            pick = free.pop(int(self._rng.integers(len(free))))
            centers[graph_id] = cells[pick]

        jitter = 0.12 * np.array(cell)
        for graph_id in graph_ids:
            offset = self._rng.uniform(-1.0, 1.0, size=2) * jitter
            centers[graph_id] = centers[graph_id] + offset
        return centers, radius

    def _add_artists(self, graph_id: str, cloud: _Cloud) -> None:
        """Create the scatter and label artists for one cloud.

        Args:
            graph_id: The graph id shown in the label.
            cloud: The cloud to attach the artists to.
        """
        cloud.scatter = self._ax.scatter([], [], s=4, linewidths=0)
        cloud.label = self._ax.text(
            0.0,
            0.0,
            f"{graph_id} ({cloud.num_points})",
            ha="center",
            va="top",
            fontsize=7,
            color="0.25",
            # Long merged names near the border must not spill onto the panels.
            clip_on=True,
        )

    def _refresh_grown_clouds(self) -> None:
        """Redraw clouds whose graphs gained nodes (e.g. after episode learning)."""
        memory = self._lm.graph_memory
        changed = False
        for graph_id, cloud in self._clouds.items():
            channels = memory.get_input_channels_in_graph(graph_id)
            if not channels:
                continue
            if memory.get_num_nodes_in_graph(graph_id, channels[0]) == cloud.num_points:
                continue
            pos = np.asarray(memory.get_locations_in_graph(graph_id, channels[0]))
            hsv = memory.get_features_by_name(graph_id, channels[0]).get("hsv")
            fresh = self._make_cloud(pos, hsv)
            fresh.center = cloud.center
            fresh.pose = cloud.pose
            fresh.angle = cloud.angle
            fresh.scatter = cloud.scatter
            fresh.label = cloud.label
            fresh.label.set_text(f"{graph_id} ({fresh.num_points})")
            fresh.place(self._radius)
            self._clouds[graph_id] = fresh
            changed = True
        if changed:
            self._draw_frame()

    def _cache_mlh_poses(self, graph_ids: list[str]) -> None:
        """Remember each object's most-likely-hypothesis rotation while it exists.

        The merge in `EvidenceGraphLM` drops the source hypotheses before this
        map sees the merged id, so the poses driving the pre-merge alignment
        must be cached ahead of time. The hypotheses are read raw rather than
        via `get_mlh_for_object`, which logs per call and rejects the
        left-handed poses merged graphs can carry.

        Args:
            graph_ids: The graph ids currently in memory.
        """
        hypotheses = getattr(self._lm, "_hypotheses", None)
        if not isinstance(hypotheses, dict):
            return
        for graph_id in graph_ids:
            graph_hyps = hypotheses.get(graph_id)
            evidence = getattr(graph_hyps, "evidence", None)
            if evidence is None or not len(evidence):
                continue
            self._mlh_poses[graph_id] = np.asarray(
                graph_hyps.poses[int(np.argmax(evidence))], dtype=np.float64
            )

    def _merge_order(self, removed: list[str], target: str) -> list[str]:
        """Recover the merge's source order from the merged graph id.

        `_merge_memory` names the merged graph `"_".join(sources)` with the
        first source providing the reference frame. Source ids may themselves
        contain underscores, so the name is peeled greedily by longest matching
        source prefix.

        Args:
            removed: The graph ids that disappeared in the merge.
            target: The merged graph id.

        Returns:
            The removed ids in merge order, or in the given order when the
            merged id doesn't parse (e.g. after a `_merged` collision suffix).
        """
        order: list[str] = []
        pool = set(removed)
        remaining = target
        while pool:
            match = max(
                (g for g in pool if remaining == g or remaining.startswith(g + "_")),
                key=len,
                default=None,
            )
            if match is None:
                return removed
            order.append(match)
            pool.discard(match)
            remaining = remaining[len(match) + 1 :]
        return order

    def _align_for_merge(self, removed: list[str], target: str) -> None:
        """Rotate the source clouds into the merged object's reference frame.

        Before the sources glide together, each one turns from its current spin
        angle to the orientation it has inside the merged graph: the first
        source (whose learned frame the merge keeps) returns upright while
        every other source takes on `R_first * R_i^-1`, the net rotation
        `_merge_memory` applied to its points. The merged cloud later fades in
        upright, matching the aligned sources.

        Args:
            removed: The source graph ids about to fade out.
            target: The merged graph id they collapse into.
        """
        order = self._merge_order(removed, target)
        first_pose = self._mlh_poses.get(order[0])
        slerps: dict[str, Slerp] = {}
        for graph_id in removed:
            pose = self._mlh_poses.get(graph_id)
            if graph_id == order[0] or first_pose is None or pose is None:
                goal = np.eye(3)
            else:
                goal = _nearest_rotation(first_pose @ pose.T)
            slerps[graph_id] = Slerp(
                (0.0, 1.0),
                Rotation.from_matrix([self._clouds[graph_id].pose, goal]),
            )
        spin_delta = math.tau * self.frame_delay / _SPIN_PERIOD_SECONDS
        for frame in range(1, self.animation_frames + 1):
            t = frame / self.animation_frames
            ease = t * t * (3.0 - 2.0 * t)
            for graph_id, cloud in self._clouds.items():
                if graph_id in slerps:
                    cloud.pose = slerps[graph_id](ease).as_matrix()
                else:
                    cloud.spin(spin_delta)
                cloud.place(self._radius)
            self._draw_frame(pause=True)

    def _animate_transition(
        self,
        snapshot: dict[str, _Cloud],
        added: list[str],
        removed: list[str],
        target: str | None,
    ) -> None:
        """Animate from the drawn state to the new memory contents.

        Removed clouds turn red and fade out — gliding toward the merge target's
        spot when there is one — while added clouds fade in at their new cell and
        the survivors glide to the relaxed layout.

        Args:
            snapshot: The new memory contents as display clouds.
            added: The graph ids present now but not drawn yet.
            removed: The drawn graph ids no longer in memory.
            target: The merged graph id the removed clouds collapse into, or
                `None` when the removal is not a merge.
        """
        previous_centers = {g: c.center for g, c in self._clouds.items()}
        centers, radius = self._layout(list(snapshot), previous_centers)

        for graph_id in added:
            cloud = snapshot[graph_id]
            cloud.center = centers[graph_id]
            self._add_artists(graph_id, cloud)

        start_centers = dict(previous_centers)
        end_centers = dict(centers)
        for graph_id in removed:
            end = centers[target] if target is not None else start_centers[graph_id]
            end_centers[graph_id] = end
        for graph_id in added:
            start_centers[graph_id] = end_centers[graph_id] = centers[graph_id]

        start_radius = self._radius
        spin_delta = math.tau * self.frame_delay / _SPIN_PERIOD_SECONDS
        for frame in range(1, self.animation_frames + 1):
            t = frame / self.animation_frames
            ease = t * t * (3.0 - 2.0 * t)
            r = start_radius + (radius - start_radius) * ease
            for graph_id, cloud in self._clouds.items():
                cloud.center = (
                    start_centers[graph_id]
                    + (end_centers[graph_id] - start_centers[graph_id]) * ease
                )
                if graph_id in removed:
                    # Merge sources hold the pose they were just aligned to.
                    cloud.place(r, alpha=1.0 - ease, shift=ease)
                else:
                    cloud.spin(spin_delta)
                    cloud.place(r)
            for graph_id in added:
                snapshot[graph_id].place(r, alpha=ease)
            self._draw_frame(pause=True)

        self._settle(snapshot, centers, radius, removed)

    def _settle(
        self,
        snapshot: dict[str, _Cloud],
        centers: dict[str, npt.NDArray[np.float64]],
        radius: float,
        removed: list[str],
    ) -> None:
        """Replace the drawn state with the new memory contents after an animation.

        Removed clouds drop their artists; surviving clouds hand their artists to
        the fresh snapshot clouds so updated geometry and counts show immediately.

        Args:
            snapshot: The new memory contents as display clouds.
            centers: The final layout center per graph id.
            radius: The final display radius.
            removed: The graph ids whose clouds finished fading out.
        """
        for graph_id in removed:
            self._clouds.pop(graph_id).remove()
        for graph_id, cloud in snapshot.items():
            drawn = self._clouds.get(graph_id)
            if drawn is not None:
                cloud.scatter = drawn.scatter
                cloud.label = drawn.label
                cloud.pose = drawn.pose
                cloud.angle = drawn.angle
                cloud.label.set_text(f"{graph_id} ({cloud.num_points})")
            cloud.center = centers[graph_id]
            cloud.place(radius)
            self._clouds[graph_id] = cloud
        self._radius = radius
        self._set_title()
        self._draw_frame()

    def _start_timer(self) -> None:
        """(Re)start the idle-spin timer on the current figure's canvas."""
        self._stop_timer()
        timer = self._fig.canvas.new_timer(interval=_SPIN_INTERVAL_MS)
        timer.add_callback(self._on_spin)
        timer.start()
        self._timer = timer

    def _stop_timer(self) -> None:
        """Stop and drop the idle-spin timer if one is running."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _on_spin(self) -> None:
        """Advance every cloud's idle rotation by one timer tick and repaint.

        Fires whenever a GUI event loop runs (including the teleop's
        wait-for-action loops), but stays out of the way while a scripted
        merge/relayout animation drives the frames itself.
        """
        if self._fig is None or self._animating or not self._clouds:
            return
        delta = math.tau * (_SPIN_INTERVAL_MS / 1000.0) / _SPIN_PERIOD_SECONDS
        for cloud in self._clouds.values():
            cloud.spin(delta)
            cloud.place(self._radius)
        self._fig.canvas.draw_idle()

    def _draw_frame(self, *, pause: bool = False) -> None:
        """Flush the canvas, optionally pausing to pace an animation frame.

        Args:
            pause: Whether to run the GUI event loop for `frame_delay` seconds so
                the frame is visible as part of an animation.
        """
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        if pause and is_interactive_backend():
            self._fig.canvas.start_event_loop(self.frame_delay)
