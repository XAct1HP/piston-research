import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from psrt import kinematics as kin
from psrt import loads as loads_mod
from psrt import thermal as thermal_mod
from psrt.margins import ComponentContext
from psrt.schema import default_state


@pytest.fixture
def state():
    return default_state("test engine")


@pytest.fixture
def geom(state):
    return kin.CrankGeometry.from_state(state)


@pytest.fixture
def sweep(state):
    return loads_mod.compute(state)


@pytest.fixture
def thermal_map(state, sweep):
    return thermal_mod.compute(state, sweep)


@pytest.fixture
def ctx(state, sweep, thermal_map):
    return ComponentContext(state, sweep, thermal_map)


def d1(f, x, h=1e-5):
    """Five-point central first derivative. Error O(h^4)."""
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)


def d2(f, x, h=1e-4):
    """Five-point central second derivative. Error O(h^4)."""
    return (-f(x + 2 * h) + 16 * f(x + h) - 30 * f(x)
            + 16 * f(x - h) - f(x - 2 * h)) / (12 * h * h)
