# imports
from contextlib import contextmanager
import pandas as pd
import os
import matplotlib.pyplot as plt

# === Display helpers ===
def show_overview(df, name):
    print(f"\n{'='*30}\n{name}\n{'='*30}")
    df.info()
    print("\nFirst rows:\n", df.head())

@contextmanager
def style_format_context(df, fmt_dict):
    """Context manager for temporary styled display (Jupyter)."""
    try:
        styled_df = df.style.format(fmt_dict)
        yield styled_df
    finally:
        pass

def format_and_display(df, fmt_dict):
    display(df.style.format(fmt_dict))



def create_geo_scatterplot(sample: pd.DataFrame):
    plot_path = os.getenv("PATH_PLOT")
    # "/home/robfra/0_Portfolio_Projekte/Road_accidents/data/plots/eda/scatter_geodata_norm.png"
    
    plt.scatter(
        sample["lon_norm"],
        sample["lat_norm"],
        s=1,
        alpha=0.3
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Accident locations (sample)")
    plt.savefig(plot_path)
    plt.show()

    return 