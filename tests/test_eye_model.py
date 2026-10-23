import numpy as np
import pytest

from eyetrack.pupil.eye_model import EyeModel

from synthetic import sphere_ellipse


def test_recovers_eyeball_center_radius_and_angles():
    center, radius = (330.0, 210.0), 120.0
    rng = np.random.default_rng(1)
    angles = rng.uniform(-30, 30, size=(200, 2))
    ellipses = [sphere_ellipse(center, radius, yw, pt) for yw, pt in angles]
    model = EyeModel.fit(ellipses, np.ones(len(ellipses)))
    assert model.cx == pytest.approx(center[0], abs=2)
    assert model.cy == pytest.approx(center[1], abs=2)
    assert model.radius == pytest.approx(radius, rel=0.03)

    px = np.array([e[0][0] for e in ellipses])
    py = np.array([e[0][1] for e in ellipses])
    yaw, pitch = model.angles_deg(px, py)
    # the generator's yaw matches the model's yaw convention exactly
    assert np.nanmax(np.abs(yaw - angles[:, 0])) < 1.5


def test_too_few_ellipses_raises():
    with pytest.raises(ValueError):
        EyeModel.fit([sphere_ellipse((0, 0), 100, 10, 0)] * 3)
