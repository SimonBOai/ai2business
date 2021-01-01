"""Test-Environment for seaborn figure style."""
from unittest import mock

from ai2business.visualization import style


@mock.patch("ai2business.visualization.style.light_mode")
def test_light_mode(light_mode):

    style.light_mode()
    assert style.light_mode.is_called


@mock.patch("ai2business.visualization.style.light_mode")
def test_light_mode_grid(light_mode):

    style.light_mode(grid=True)
    assert style.light_mode.is_called


@mock.patch("ai2business.visualization.style.dark_mode")
def test_dark_mode(dark_mode):

    style.dark_mode()
    assert style.dark_mode.is_called


@mock.patch("ai2business.visualization.style.dark_mode")
def test_dark_mode_grid(dark_mode):

    style.dark_mode(grid=True)
    assert style.dark_mode.is_called
