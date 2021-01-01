"""Theme mode for seaborn."""
import seaborn as sns


def light_mode(grid: bool = False):
    """Set the seaborn theme style to light mode.

    Args:
        grid (bool, optional): Activate grid mode for plot. Defaults to False.
    """
    if grid:
        sns.set_theme(style="darkgrid")
    else:
        sns.set_theme(style="dark")


def dark_mode(grid: bool = False):
    """Set the seaborn theme style to dark mode.

    Args:
        grid (bool, optional): Activate grid mode for plot. Defaults to False.
    """
    if grid:
        sns.set_theme(style="whitegrid")
    else:
        sns.set_theme(style="white")
