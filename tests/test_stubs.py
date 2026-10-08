import numpy as np
import pytest

from flycraft import eye
from flycraft.brain.control import Command, Controller, max_turn_deg
from flycraft.brain.stubs import STUBS, Oracle, RandomWalker, object_azimuth
from flycraft.config import DecoderConfig

DEC = DecoderConfig()
LIM = 180.0 * 8 / 22.4


def test_max_turn():
  assert max_turn_deg(DEC) == pytest.approx(LIM)


@pytest.mark.parametrize("polarity", sorted(eye.POLARITY))
@pytest.mark.parametrize("az", [0.0, 30.0, -45.0, 90.0, -120.0, 177.5, -177.5, 180.0])
def test_object_azimuth_finds_the_disk(az, polarity):
  for radius in (2.0, 10.0, 60.0):
    got = object_azimuth(eye.disk(az, radius, polarity), polarity)
    assert eye.wrap_deg(got - az) == pytest.approx(0.0, abs=1.0), radius


def test_object_azimuth_sees_nothing_in_a_blank_scene():
  assert object_azimuth(eye.blank("dark_on_bright"), "dark_on_bright") is None


def test_oracle_turns_toward_the_object_within_the_limit():
  oracle = Oracle(DEC, "dark_on_bright")
  near = oracle.step(eye.disk(20.0, 5.0, "dark_on_bright"), 0.0)
  assert near.dtheta == pytest.approx(20.0, abs=1.0) and near.speed == 1.0
  far = oracle.step(eye.disk(-150.0, 5.0, "dark_on_bright"), 0.0)
  assert far.dtheta == pytest.approx(-LIM)
  assert far.turn_raw == pytest.approx(-150.0, abs=1.0)
  assert oracle.step(eye.blank("dark_on_bright"), 0.0) == Command(0.0, 1.0)


def test_oracle_reads_the_polarity_it_announces():
  oracle = Oracle(DEC, "bright_on_dark")
  assert oracle.identity.polarity == "bright_on_dark"
  assert oracle.step(eye.disk(45.0, 5.0, "bright_on_dark"), 0.0).dtheta == pytest.approx(
    45.0, abs=1.0)


def test_random_walker_is_seeded_per_episode_and_walks_at_s0():
  a, b = RandomWalker(DEC, "dark_on_bright"), RandomWalker(DEC, "dark_on_bright")
  img = eye.blank("dark_on_bright")
  a.start_episode(0, 11, "train")
  b.start_episode(5, 11, "eval")
  run_a = [a.step(img, 0.0) for _ in range(50)]
  assert run_a == [b.step(img, 0.0) for _ in range(50)]
  a.start_episode(1, 12, "train")
  assert [a.step(img, 0.0) for _ in range(50)] != run_a
  assert all(abs(c.dtheta) <= LIM and c.speed == DEC.s0 for c in run_a)
  assert np.ptp([c.dtheta for c in run_a]) > LIM  # it really does turn both ways


def test_stub_identities():
  for name, cls in STUBS.items():
    stub = cls(DEC, "dark_on_bright")
    assert isinstance(stub, Controller)
    assert stub.identity.brain_id == f"stub-{name}"
    assert stub.identity.decision_frames == DEC.decision_frames
    assert stub.identity.plasticity is False
