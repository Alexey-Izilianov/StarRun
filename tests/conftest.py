from __future__ import annotations

import numpy as np
import pytest

from starrun.balance import Balance


@pytest.fixture(scope="session")
def balance() -> Balance:
    return Balance.load()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
