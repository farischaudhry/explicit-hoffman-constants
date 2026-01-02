import matplotlib.pyplot as plt

def set_publication_style():
    """Sets up standard plotting style."""
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.figsize': (10, 7),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.5,
    })
