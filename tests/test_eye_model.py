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


def test_fixed_radius_keeps_the_fitted_center():
    center = (330.0, 210.0)
    rng = np.random.default_rng(2)
    ellipses = [sphere_ellipse(center, 120.0, yw, pt) for yw, pt in rng.uniform(-30, 30, size=(200, 2))]
    model = EyeModel.fit(ellipses, np.ones(len(ellipses)), radius=150.0)
    assert model.radius == 150.0 and model.radius_source == "fixed"
    assert model.cx == pytest.approx(center[0], abs=2) and model.cy == pytest.approx(center[1], abs=2)
    assert EyeModel(**model.to_dict()).radius_source == "fixed"     # round-trips for the live worker


def test_radius_warning_flags_implausible_scales():
    from eyetrack.pupil.eye_model import radius_warning
    assert radius_warning(53, 52) is not None and "small" in radius_warning(53, 52)   # the 2026-11-02 session
    assert radius_warning(130, 52) is None
    assert "large" in radius_warning(400, 52)
