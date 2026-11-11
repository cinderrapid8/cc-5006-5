"""Scientific experiment video overlay tool.

Burns synthetic sensor curves, current readings and event markers into
experiment video. Time alignment between data and picture is expressed
through manual anchors with an explicit uncertainty range.
"""

__all__ = ["model", "timeline", "resample", "render", "store"]
