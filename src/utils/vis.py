'''
2025-05-01
Author: Dan Schumacher
How to run:
   python ./src/utils/vis/vis.py
'''

import base64
import datetime
import os, re, sys
sys.path.append("./src")
import matplotlib.pyplot as plt
import numpy as np
from utils.file_io import load_tsdata_list
from utils.enums_dcs import TSData

 # Function to images for prompting
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def plot_time_series(obs: 'TSData', idx: int = 0):
    """
    Plots a univariate time series and saves it to ./figures/{idx}_tiny.png.
    Handles metadata like time axis and units cleanly.
    """
    # Prepare x and y
    y = obs.series
    x = np.arange(len(y))

    # Try to construct datetime x-axis from metadata
    start_date = obs.metadata.get("start", "")
    freq = obs.metadata.get("frequency", "daily").lower()
    x_labels = None

    if start_date:
        try:
            dt_start = datetime.strptime(start_date, "%d-%m-%Y")
            if freq == "daily":
                x_labels = [dt_start + datetime.timedelta(days=i) for i in range(len(y))]
            elif freq == "monthly":
                x_labels = [dt_start + datetime.timedelta(days=i * 30) for i in range(len(y))]
            # Add more rules if needed
        except Exception:
            x_labels = None

    # Axis labels and title
    y_lab = obs.metadata.get("units", "Value")
    title = obs.description_tiny.strip() if obs.description_tiny else "Time Series"

    # Plot
    plt.figure(figsize=(4, 3), dpi=80)
    plt.plot(x, y, linewidth=0.8)

    plt.title(title, fontsize=8)
    plt.xlabel("Time" if x_labels else "t", fontsize=6)
    plt.ylabel(y_lab, fontsize=6)

    # Optional: show some formatted dates
    if x_labels and len(x_labels) <= 1000:
        xticks_idx = np.linspace(0, len(x_labels) - 1, num=5, dtype=int)
        xticks = [x_labels[i].strftime("%Y-%m-%d") for i in xticks_idx]
        plt.xticks(xticks_idx, xticks, rotation=20, fontsize=5)
    else:
        plt.xticks(fontsize=5)

    plt.yticks(fontsize=5)
    plt.tight_layout(pad=0.2)
    os.makedirs("./figures", exist_ok=True)
    plt.savefig(f"./figures/{idx}_tiny.png", bbox_inches='tight', dpi=80)
    plt.close()
