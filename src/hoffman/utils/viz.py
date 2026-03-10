import matplotlib.pyplot as plt

def set_plotting_style():
    """Sets up standard plotting style."""
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.figsize': (10, 7),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.5,
    })
