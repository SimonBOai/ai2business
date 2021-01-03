"""Theme mode for seaborn."""
import seaborn as sns


def light_mode(grid: bool = False) -> sns.set_theme:
    """Set the seaborn theme style to light mode.

    Args:
        grid (bool, optional): Activate grid mode for plot. Defaults to False.

    Returns:
        sns.set_theme: Environment setting function of seaborn.
    """
    if grid:
        return sns.set_theme(style="whitegrid")
    sns.set_theme(style="white")


def dark_mode(grid: bool = False) -> sns.set_theme:
    """Set the seaborn theme style to dark mode.

    Args:
        grid (bool, optional): Activate grid mode for plot. Defaults to False.

    Returns:
        sns.set_theme: Environment setting function of seaborn.
    """
    if grid:
        return sns.set_theme(style="darkgrid")
    return sns.set_theme(style="dark")
