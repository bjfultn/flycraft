# flycraft M3c: zerglings against marines

- Date: 2026-10-08
- Parent: [the M3b demo](2026-10-08-flycraft-m3b-defeatroaches.md)
- Status: a demo, not an experiment, like M3b. Nothing here is pre-registered.

## 1. Goal

The fly commands Zerg, not Terran: a zergling squad against marines. A fly is closer to a swarm
of small melee biters than to a rifle squad, and melee units cannot shoot each other by
accident. It is also the first map flycraft makes itself, the step toward a map where a person
plays the marines against the fly.

## 2. The minigame

DefeatMarines is DefeatRoaches with the units swapped: 9 zerglings start against 4 marines, and
each time all 4 marines die, 4 new marines and 5 more zerglings arrive. Each marine killed
scores +10, each zergling lost -1, and an episode ends at 120 seconds or when every zergling is
dead. The player is Zerg.

flycraft does not ship SC2's maps, so it builds this one from the player's own copy of
DefeatRoaches: `flycraft-maps` reads `Maps/mini_games/DefeatRoaches.SC2Map` and writes
`Maps/flycraft/DefeatMarines.SC2Map` under the SC2 install. It changes four things:

- **The map script** (`MapScript.galaxy`, what SC2 runs): every `"Marine"` becomes
  `"Zergling"` and every `"Roach"` becomes `"Marine"`. That covers the squad, the waves and the
  check that carries surviving squad units into the next wave.
- **The preload list** (`Preload.xml`): the same swap, so SC2 loads the zergling's art with
  the map.
- **The players' races** (`MapInfo`): player 1 becomes Zerg and player 2 Terran.
- **The strings** (`GameStrings.txt`): the map's name, its victory text and player 2's name.

The editor's copy of the triggers (`Triggers`) is left as it was, so the map opened in the SC2
editor still reads as DefeatRoaches; SC2 runs the script, not the editor's copy. The map is an
MPQ archive. pysc2's MPQ library (mpyq) only reads, so flycraft writes it with its own small
writer (classic format 1 tables, each file stored as one zlib unit, as Blizzard stores this
map's files) rather than depend on a native MPQ library.

A spike ran the built map in SC2 before any of this was written: 9 zerglings (unit type 105)
against 4 marines (48), the race set to Zerg, and pysc2's scripted agent scoring 319 and 329
in two episodes, against 0 for a squad that does nothing.

## 3. What changes

Nothing in the brain, the protocol, the eye or the body. A task now names its race, and
DefeatMarines is DefeatRoaches' task with race Zerg: every marine is a target drawn as its own
disk, the squad attacks the marine nearest the heading within 30 degrees, and otherwise
attack-moves a step along it onto clear ground. pysc2's scripted agent for it is the
DefeatRoaches one (attack the lowest enemy pixel).

## 4. Unit counts

The counts are DefeatRoaches' unless the baselines show they leave no room: if the random
walker scores within 20% of the scripted agent's mean, more marines start each wave, set on the
baselines alone and fixed before the fly runs. Any change is disclosed here and in the video.

**Ruling (2026-10-08):** the random walker scored 129.6, 40% of the scripted agent's 322.9, so
the counts stay DefeatRoaches': 9 zerglings against 4 marines a wave.

## 5. Baselines and the demo

The M3b set, on `--map DefeatMarines`, train mode, seed 1, 20 episodes each: scripted, random,
oracle and the real fly (vpn rung, w_scale 0.84375, untrained, plasticity off). Then a
watch-mode run of the real fly with the brain view, recorded, at `--face 60` as in M3b (and said
in the caption).

**Results (2026-10-08, 20 episodes each):**

| Pilot | Mean | Median | SD |
|---|---|---|---|
| scripted (pysc2's agent) | 322.9 | 329.5 | 64.9 |
| oracle (same body) | 206.9 | 198.5 | 123.2 |
| random walker | 129.6 | 143.0 | 60.6 |
| real fly | 61.5 | 62.5 | 58.1 |

As in M3b, the fly's score splits on whether it turns. In 7 episodes it made no turn at all and
scored 0; in the other 13 it turned and fought, averaging 94.6. Zerglings told to attack-move
fight whatever marines they meet, so a random walker that keeps changing direction finds the
marines more often than a fly that does not turn.

## 6. Tests

The writer: MPQ's hash of its table names matches the published values, and pysc2's own MPQ
reader (mpyq) reads back every file of an archive it wrote, empty, tiny, incompressible, large
and with colliding hash slots. The build: the swaps land in a small made-up map and nowhere
else, a source that lacks any of them stops the build with a message naming what is missing,
and a missing DefeatRoaches map says where it was looked for. The game side: the task is Zerg,
SC2Game asks pysc2 for the task's race, pysc2 finds the map under `flycraft/`, and a missing
built map says to run `flycraft-maps`.
