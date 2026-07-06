"""Deterministic episode recording and scientific replay rendering."""

from .recorder import ReplayBundle, ReplayRecorder, record_episode
from .render import render_frame_2d, render_frame_3d, render_video


def export_paraview(*args, **kwargs):
    """Load the optional VTK exporter only when it is requested."""

    from aeolus.viewer.export import export_paraview as export

    return export(*args, **kwargs)


__all__ = [
    "ReplayBundle",
    "ReplayRecorder",
    "record_episode",
    "render_frame_2d",
    "render_frame_3d",
    "render_video",
    "export_paraview",
]
