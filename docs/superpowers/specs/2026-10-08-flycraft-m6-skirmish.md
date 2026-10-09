# flycraft M6: a person plays the fly

- Date: 2026-10-08
- Parent: [the design spec, section 15](2026-10-07-flycraft-design.md), after
  [M3c](2026-10-08-flycraft-m3c-defeatmarines.md)
- Status: a demo, not an experiment. Nothing here is pre-registered.

## 1. Goal

A person plays a match against the fly. The person commands 4 marines; the fly commands 9
zerglings with the M3c reflex (the squad attacks the marine nearest its heading within 30
degrees, else attack-moves along it). Both play in real time on one PC: two SC2 processes, one
game over LAN, the way pysc2's `play_vs_agent` joins a human and an agent. Exit: the person
plays a whole match against the fly and both sides see the result.

## 2. The map: Skirmish

`flycraft-maps` builds `Maps/flycraft/Skirmish.SC2Map` next to DefeatMarines, from the same
DefeatRoaches source. It changes:

- **The players** (`MapInfo`): player 2 becomes a User slot, not a Computer, so a second
  client can join it. Races stay as DefeatRoaches has them: player 1 Terran, player 2 Zerg.
- **The preload list** (`Preload.xml`): `Roach` becomes `Zergling`.
- **The name** (`GameStrings.txt` and `DocumentHeader`, whose copy is the one SC2 shows):
  "Skirmish", and player 2 "Fly".
- **The map script** (`MapScript.galaxy`): replaced whole by flycraft's own,
  `flycraft/game/skirmish.galaxy`, written for this map and shipped with the package (MIT).
  It keeps nothing of Blizzard's script but the arena's regions (1 the playable area, 4 and 5
  the two start spots).

A spike ran a two-player DefeatMarines before this was written: with player 2 set to User,
the creating client got player 1 and the joining one player 2, raw moves reached both squads,
and the game never ended by itself (nothing called `GameOver`). The joining side saw fog,
because the minigame reveals the arena to player 1 only. The script below fixes both.

## 3. The rules (the script)

- Both players see the whole arena, with the camera fixed on it as in the minigames.
- **A round**: every unit is removed; 4 marines (player 1) and 9 zerglings (player 2) appear
  at the two start spots, sides picked at random; each squad starts selected for its player.
  The round ends when one squad is dead (that side loses the round), or after 120 seconds
  of game time (a draw).
- Between rounds the units freeze and the round's result shows for 3 seconds of game time
  (about 2 real seconds at Faster speed). The next round's squads then appear frozen for 3
  more, under "Round 4 (Marines 2, fly 1)",
  and the round starts. The score also stays on screen as an objective line.
- **The match**: first to 3 round wins. After 7 rounds the side with more wins takes it, and
  equal wins is a tie. The script then ends the game for both players with `GameOver`
  (victory, defeat or tie), which SC2 reports to both clients.

## 4. Running a match

- **The person** runs `flycraft-match` (new): it starts SC2 visible, creates a realtime
  Skirmish game for two, joins it as player 1 (Terran), and waits for the fly on a local port
  (14380 and the 4 after it). When the game ends it prints one JSON line with the result,
  leaves the end screen up for 10 seconds, and closes. It mirrors `play_vs_agent`'s host,
  except that it joins before the fly connects, so the person is always player 1. Flags:
  `--port` (14380), `--name` (the person's name in the game), `--window` and `--wait` (how
  long to wait for the fly, 300 seconds).
- **The fly**: `flycraft-client --brain URL --map Skirmish --join 14380` joins as player 2
  through pysc2's `LanSC2Env` in real time, with its SC2 window minimized. A match is one
  episode for the brain, so the heading carries over between rounds (`--face` sets it once,
  at the start). It plays exactly one match and does not relaunch a failed game. Its JSON
  line adds `outcome`: 1 the fly won, -1 it lost, 0 a tie.
- Either side checks its player number on the first observation and stops with a message if
  it is wrong.
- **Real time**: the fly decides by the game's clock, once 8 game frames (357 ms at 22.4
  frames a second) have passed since its last decision, not on every 8th observation. In real
  time the game does not wait for the brain, which took about 430 ms a decision in M3, so a
  decision lands as soon as the brain is ready: about 430 ms apart, where counting
  observations would make it about 740 ms. Off real time the two rules are the same. The fly
  acts late by about one decision; the video says so.

## 5. What changes

Nothing in the brain, the protocol, the eye or the body. New: the Skirmish build in
`flycraft-maps`, the script, a `Skirmish` map class for pysc2 (2 players), a `Skirmish` task
(DefeatMarines' task: Zerg, the 30 degree cone, and a `versus` flag, so the client will not
play it without `--join`), `flycraft-match`, and the client's `--join`. `Frame` and the result
record gain `outcome`. The pilot's decision rule moves to the game's clock (section 4).

## 6. Tests

The build: the control byte flips only for player 2, a source whose player 2 is not a Computer
or not Zerg stops the build, the preload swap and the rename land, and the script in the built
map is flycraft's file, byte for byte. The script: it names regions 1, 4 and 5, both players,
`GameOver`, and the round and match limits. The host: the ports it asks for, the join order
(its join is under way before it accepts the fly), a busy port, a fly that never joins, a port
it cannot listen on, the person landing as player 2, a game with no result for the person,
and the result line. The client: `--join` needs the Skirmish map, takes no `--episodes` above
1 and no train mode, and implies watch mode in real time; Skirmish without `--join` stops
with a message; a match plays once with no relaunch; the fly must be player 2; a failed join
names the port; the last frame's reward becomes the outcome; in real time the pilot decides
by the game's clock. The game itself is
checked on SC2 before the person plays: a scripted stand-in for the person against the fly,
then a match.
