import numpy as np
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


@pytest.mark.parametrize("target, bearing", [((60.0, 40.0), 0.0), ((40.0, 20.0), -90.0),
                                             ((20.0, 40.0), 180.0)])
def test_facing_a_target_points_within_the_cone(target, bearing):
  body = Body()
  body.face((40.0, 40.0), target, 0.0, seed=5)
  assert abs(body.heading - bearing) < 1e-9
  offsets = []
  for seed in range(50):
    body.face((40.0, 40.0), target, 60.0, seed)
    offsets.append((body.heading - bearing + 180.0) % 360.0 - 180.0)
  assert all(-60.0 <= o <= 60.0 for o in offsets) and max(offsets) - min(offsets) > 60.0
  body.face((40.0, 40.0), target, 60.0, 7)
  again = body.heading
  body.face((40.0, 40.0), target, 60.0, 7)
  assert body.heading == again  # fixed by the seed


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


def test_the_order_a_step_ahead_is_the_tasks():
  body = Body(BodyParams(order="attack"))
  assert body.motor((40.0, 40.0), True, 1.0) == ("attack", (46.0, 40.0))
  assert body.motor((40.0, 40.0), True, 0.0) == STOP
  assert body.motor((40.0, 40.0), False, 1.0) == SELECT


def _squad(*boxes):
  own = np.zeros((84, 84), bool)
  for y0, y1, x0, x1 in boxes:
    own[y0:y1, x0:x1] = True
  return own


def test_an_attack_step_goes_past_the_squad():
  # SC2 takes an attack aimed at one of the player's own marines as an order to shoot it
  body = Body(BodyParams(order="attack"))
  own = _squad((36, 45, 36, 45))  # marines out to x = 44 around the middle (40, 40)
  act = body.motor((40.0, 40.0), True, 0.5, own=own)  # the step lands at (43, 40), on them
  assert act == ("attack", (46.0, 40.0))  # the first pixel 2 px clear of x = 44
  body.heading = 90.0
  assert body.motor((40.0, 40.0), True, 0.5, own=own) == ("attack", (40.0, 46.0))


def test_an_attack_step_on_clear_ground_is_not_moved():
  body = Body(BodyParams(order="attack"))
  own = _squad((39, 42, 39, 42))
  assert body.motor((40.0, 40.0), True, 1.0, own=own) == ("attack", (46.0, 40.0))


def test_an_attack_step_with_no_clear_ground_ahead_is_a_move():
  body = Body(BodyParams(order="attack"))
  own = _squad((36, 45, 36, 84))  # marines all the way to the right edge
  assert body.motor((40.0, 40.0), True, 1.0, own=own) == ("move", (46.0, 40.0))
  body.heading = 45.0
  assert body.motor((82.0, 82.0), True, 1.0, own=_squad((80, 84, 80, 84))).kind == "move"


def test_a_move_step_ignores_the_squad():
  body = Body()
  own = _squad((36, 45, 36, 45))
  assert body.motor((40.0, 40.0), True, 0.5, own=own) == ("move", (43.0, 40.0))


def test_a_strike_attacks_the_target_at_any_speed():
  body = Body(BodyParams(order="attack"))
  for speed in (1.0, 0.0):
    assert body.motor((40.0, 40.0), True, speed, strike=(70.0, 20.0)) == ("attack", (70.0, 20.0))
  assert body.motor((40.0, 40.0), False, 1.0, strike=(70.0, 20.0)) == SELECT
  assert body.motor(None, True, 1.0, strike=(70.0, 20.0)) == NOOP


def test_an_unseen_marine_heads_for_the_strike():
  body = Body()
  body.locate((40.0, 40.0))
  body.motor((40.0, 40.0), True, 1.0, strike=(80.0, 40.0))
  assert body.locate(None) == (40.0 + body.params.stride_px, 40.0)



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
