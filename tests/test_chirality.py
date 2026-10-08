"""End-to-end chirality, code chain (spec sections 7.6 and 12).

Screen frame: x right, y down. A marine facing screen-up has heading -90 degrees. Every
expected value below comes from that fixed geometry, not from flycraft code: the test runs
the game's own render and body code and checks them against literals.
"""

import numpy as np
import pytest

from flycraft import eye
from flycraft.brain.decoder import Calibration, Decoder, resolve_groups
from flycraft.brain.retina import Retina, RetinaGeometry
from flycraft.config import DecoderConfig, EyeConfig, RetinaConfig
from flycraft.game import render
from flycraft.game.body import Body

MARINE = (42.0, 42.0)
HEADING = -90.0  # screen-up


@pytest.fixture(scope="module")
def retina(conn):
  return Retina(RetinaGeometry.build(conn), RetinaConfig(), EyeConfig())


@pytest.mark.parametrize("beacon, expected_az, side", [
  ((60.0, 42.0), 90.0, "R"),  # screen-right of a screen-up marine is the fly's right
  ((24.0, 42.0), -90.0, "L"),
])
def test_chirality_code_chain(conn, retina, beacon, expected_az, side):
  # 1. the screen position becomes an egocentric azimuth
  az, _ = render.egocentric(MARINE, beacon, HEADING)
  assert az == pytest.approx(expected_az)
  # 2. the rendered disk lands in that half of the eye image
  img = render.render_eye(MARINE, render.Blob(beacon, 28), HEADING, "dark_on_bright")
  half = {"L": img[:, : eye.EYE_COLS // 2].mean(), "R": img[:, eye.EYE_COLS // 2:].mean()}
  other = "L" if side == "R" else "R"
  assert half[side] < half[other]
  # 3. that eye's photoreceptors see a darker image
  g = retina.geom
  lum = retina.sample(img)
  pr_side = g.pr_side[g.assigned]
  assert lum[pr_side == side].mean() < lum[pr_side == other].mean()
  # 4. DNa02 on that side firing more turns the fly toward it
  cfg = DecoderConfig()
  groups = resolve_groups(conn, cfg)
  d = Decoder(groups, cfg, Calibration(k_turn=18.0, turn_silent=False))
  hot = groups.turn_pos[0] if side == "R" else groups.turn_neg[0]
  spikes = np.zeros((10_000, d.watch.size), bool)
  spikes[::200, np.searchsorted(d.watch, hot)] = True
  d.observe(spikes)
  dtheta, _ = d.command()
  assert np.sign(dtheta) == np.sign(expected_az)
  # 5. the body turns, and the next move target swings toward the beacon's side of the screen
  body = Body()
  body.heading = HEADING
  body.turn(dtheta)
  x, y = body.motor(MARINE, True, 1.0).xy
  assert (x > MARINE[0]) if side == "R" else (x < MARINE[0])
  assert y < MARINE[1]  # still ahead: a clockwise turn from screen-up, not a reversal
