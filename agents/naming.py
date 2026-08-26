"""Model checkpoint naming convention shared with the dashboard.

Pattern: models/agent16_{W}x{H}_{M}_r{R}.zip
  W x H : board size
  M     : mine count
  R     : training round within that board config
Example: models/agent16_16x16_20_r4.zip
"""


def model_path(width, height, mines, round_num):
    return (f"models/agent16_{width}x{height}_{mines}"
            f"m_r{round_num}.zip")


def model_glob(width=None, height=None, mines=None):
    w = width if width is not None else "*"
    h = height if height is not None else "*"
    m = f"{mines}m" if mines is not None else "*"
    return f"models/agent16_{w}x{h}_{m}_r*.zip"
