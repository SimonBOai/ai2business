"""Test-Environment for seaborn figure style."""

from ai2business.visualization import style

import seaborn as sns


def test_light_mode():

    assert style.light_mode() == sns.set_theme(style="white")


def test_light_mode_grid():

    assert style.light_mode(grid=True) == sns.set_theme(style="whitegrid")


def test_dark_mode():

    assert style.dark_mode() == sns.set_theme(style="dark")


def test_dark_mode_grid():

    assert style.dark_mode(grid=True) == sns.set_theme(style="darkgrid")
