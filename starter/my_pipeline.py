"""v1 pipeline: a possession state machine instead of a majority vote.

    python starter/my_pipeline.py                 # writes evaluation/results/my_pipeline.csv
    python evaluation/validate_submission.py evaluation/results/my_pipeline.csv
    python evaluation/run_evaluation.py

Idea. The baseline gives every frame its own vote and then smooths, and a frame with no ball becomes D, which
splits real possessions in two. Here a possession is a STATE that persists:

  * A frame gives evidence for a team only when the ball is seen AND a player is within EVIDENCE_RADIUS_M of it.
    A ball in flight, or one nobody is near, says nothing.
  * The state changes to the other team only after COMMIT_FRAMES frames of evidence for it, with no evidence
    for the current team in between. This is the "keeps the ball for a moment" proxy from CHALLENGE.md.
  * When it commits, the change is back-dated to where the new evidence began, so timing is not late.
  * Frames with no ball or no evidence HOLD the current state. On the ranking metric that costs nothing
    inside a possession, and unlike the baseline it never invents a dead-ball stretch.
  * Only the two graded halves are labelled; everything else is X.

Not done yet (on purpose, in this order): dead-ball D detection, Smart Tagging restarts, ball gap filling.
The constants below are first guesses. Do not tune them on the public 5 minutes (RULES.md) -- reason about
them, or check them on other games.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_of_each_track, write_submission, HOME, AWAY  # noqa: E402

EVIDENCE_RADIUS_M = 3.0   # a player this close to the ball counts as having it
COMMIT_FRAMES = 40        # evidence frames needed to switch team (25 frames = 1 s)
EXPIRE_FRAMES = 200       # a pending switch with no new evidence for this long is dropped
OUT_NAME = "my_pipeline"


def read_tracking(raw_path, n_frames, pitch, n_slots):
    """Ball (n, 2) and players (n, n_slots, 2) in metres, NaN where missing."""
    length, width = pitch
    ball = np.full((n_frames, 2), np.nan, np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1  # frame_id counts from 1
            if parts[2][0] != "N":
                b = parts[2].split(",", 2)
                ball[i] = (float(b[0]) * length, float(b[1]) * width)
            for slot, group in enumerate(parts[1].split(";")):
                if slot >= n_slots or group[0] == "N":
                    continue
                g = group.split(",", 2)
                players[i, slot] = (float(g[0]) * length, float(g[1]) * width)
    return ball, players


def evidence(ball, players, team_of_track):
    """Per frame: 0 home, 1 away, -1 no evidence, using the nearest tracked player within the radius."""
    d = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    d_home = d[:, team_of_track == HOME].min(axis=1)
    d_away = d[:, team_of_track == AWAY].min(axis=1)
    nearest = np.minimum(d_home, d_away)
    team = np.where(d_home <= d_away, HOME, AWAY)
    return np.where(nearest <= EVIDENCE_RADIUS_M, team, -1).astype(np.int8)


def label_frames(ev, blocks, commit=COMMIT_FRAMES, expire=EXPIRE_FRAMES):
    """The state machine. `ev` is the per-frame evidence; `blocks` the graded (half, first, last) ranges."""
    labels = np.full(len(ev), "X", dtype="<U1")
    letter = {HOME: "H", AWAY: "A"}
    for _, first, last in blocks:
        state, cand, count, cand_start, last_seen = None, None, 0, 0, 0
        for f in range(first, last + 1):
            t = int(ev[f])
            if t >= 0:
                if state is None:
                    state = t
                    labels[first:f] = letter[t]  # head of the half: back-fill with the first team seen
                elif t == state:
                    cand, count = None, 0
                else:
                    if cand != t:
                        cand, count, cand_start = t, 0, f
                    count += 1
                    last_seen = f
                    if count >= commit:
                        state = t
                        labels[cand_start:f] = letter[t]  # back-date to where the new evidence began
                        cand, count = None, 0
            if cand is not None and f - last_seen > expire:
                cand, count = None, 0
            if state is not None:
                labels[f] = letter[state]
    return labels


def n_possessions(labels):
    runs = [l for l in labels if l in "HA"]
    return sum(1 for i, l in enumerate(runs) if i == 0 or l != runs[i - 1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    args = parser.parse_args()

    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, None)
    n_frames = source_frames("tactical", GAME)
    team_of_track = team_of_each_track(meta_path)
    print("reading tracking ...")
    ball, players = read_tracking(raw_path, n_frames, GAMES[GAME]["pitch"], len(team_of_track))
    ev = evidence(ball, players, team_of_track)
    blocks = half_video_blocks(GAME)
    graded = np.zeros(n_frames, bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    print("graded frames: ball seen {:.1f}%, with evidence {:.1f}%".format(
        100 * np.mean(~np.isnan(ball[graded, 0])), 100 * np.mean(ev[graded] >= 0)))

    labels = label_frames(ev, blocks)
    path = write_submission(os.path.join(args.out, OUT_NAME + ".csv"), labels)
    print("possessions in my file: {}".format(n_possessions(labels[graded])))
    print("written", os.path.relpath(path, ROOT))
