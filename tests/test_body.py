import pytest

from flycraft.config import DecoderConfig
from flycraft.game.body import NOOP, SELECT, STOP, Body, BodyParams


def test_defaults_match_the_decoder():
  dec = DecoderConfig()
  p = BodyParams()
  for name in ("max_turn_deg_s", "decision_frames", "game_fps"):
    assert getattr(p, name) == getattr(dec, name), name


def test_max_turn_is_180_deg_per_game_second():
  assert BodyParams().max_turn_deg == pytest.approx(180.0 * 8 / 22.4)  # 64.29 degrees


def test_reset_is_seeded():
  a, b, c = Body(), Body(), Body()
  a.reset(7)
  b.reset(7)
  c.reset(8)
  assert a.heading == b.heading != c.heading
  assert -180.0 <= a.heading < 180.0


def test_turn_integrates_clips_and_wraps():
  body = Body()
  body.heading = 170.0
  assert body.turn(20.0) == 20.0
  assert body.heading == pytest.approx(-170.0)  # wrapped
  lim = BodyParams().max_turn_deg
  assert body.turn(500.0) == pytest.approx(lim)
  assert body.heading == pytest.approx(-170.0 + lim)
  assert body.turn(-500.0) == pytest.approx(-lim)
  assert body.heading == pytest.approx(-170.0)


@pytest.mark.parametrize("heading, target", [
  (0.0, (46.0, 40.0)),
  (90.0, (40.0, 46.0)),  # positive heading points screen-down
  (-90.0, (40.0, 34.0)),
  (180.0, (34.0, 40.0)),
])
def test_move_target_is_one_step_along_the_heading(heading, target):
  body = Body()
  body.heading = heading
  act = body.motor((40.0, 40.0), True, 1.0)
  assert act.kind == "move"
  assert act.xy == pytest.approx(target, abs=1e-9)


def test_speed_scales_the_step():
  body = Body()
  assert body.motor((40.0, 40.0), True, 0.5).xy == pytest.approx((43.0, 40.0))


def test_move_target_is_clipped_to_the_screen():
  body = Body(BodyParams(step_px=20.0))
  body.heading = -135.0
  assert body.motor((5.0, 3.0), True, 1.0).xy == (0.0, 0.0)
  body.heading = 45.0
  assert body.motor((80.0, 82.0), True, 1.0).xy == (83.0, 83.0)


def test_slow_means_stop_and_unselected_means_select():
  body = Body()
  assert body.motor((40.0, 40.0), True, 0.049) == STOP
  assert body.motor((40.0, 40.0), True, 0.05).kind == "move"
  assert body.motor((40.0, 40.0), False, 1.0) == SELECT
  assert body.motor(None, True, 1.0) == NOOP



# Where the marine is (spec 7.1): SC2 draws the beacon over it, and scores only well inside

def test_a_seen_marine_is_where_it_is_seen():
  body = Body()
  assert body.locate(None) is None  # never seen
  assert body.locate((40.0, 40.0)) == (40.0, 40.0)
  assert body.locate((41.5, 39.0)) == (41.5, 39.0)


def test_a_hidden_marine_is_dead_reckoned_toward_its_last_order():
  body = Body()  # stride 4 px a decision: L = 6 px is 1.5 decisions of travel
  body.locate((40.0, 40.0))
  body.motor((40.0, 40.0), True, 1.0)  # heading 0: sent to (46, 40)
  assert body.locate(None) == pytest.approx((44.0, 40.0))  # one stride of the six
  body.motor((44.0, 40.0), True, 0.5)  # sent to (47, 40), within a stride
  assert body.locate(None) == pytest.approx((47.0, 40.0))
  body.motor((47.0, 40.0), True, 0.0)  # stopped
  assert body.locate(None) == pytest.approx((47.0, 40.0))
  assert body.locate(None) == pytest.approx((47.0, 40.0))  # no order, no travel


def test_reset_forgets_where_the_marine_was():
  body = Body()
  body.locate((40.0, 40.0))
  body.motor((40.0, 40.0), True, 1.0)
  body.reset(3)
  assert body.locate(None) is None
