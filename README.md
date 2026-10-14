# Aurora-X pro

**A telescope-survey scheduling agent in Rust — the final version submitted by team
Aurora-X to the [GOSIM 2026 Agentic Observer Hackathon](https://create.gosim.org/survey26/)
(survey26).**

This repository is an honest record of a small, careful submission. The agent is the
organizers' `rust-pro` example with a **21-line patch across three source files** — every
change is a mechanism fix, none is tuned to a specific task card. It finished **30th of
102 ranked teams** on the hidden final cards, ahead of both official example baselines.

The repository contains the code exactly as submitted (platform revision
`15dfec80-0d76-49ca-acfa-df35c8e24c99`, project name `Aurora-X pro-v4-reqfloor`).

## What the agent does

Each run is one simulated observing season on a multi-fibre spectroscopic telescope. The
engine sends a decision request for every step; the agent answers `observe` (pointing,
fibre-to-target assignment, exposure time, program), `wait`, `report` (an instrument
fault) or `finish`. The score combines the science value of each target's best exposure
with program-band multipliers, penalties for required targets left unobserved, rewards
for paid observation requests completed before their deadline, fault reports (correct
reports earn points; false ones are penalised after two free), and a uniformity term.

## Results

**Final ranking: 30th of 102 ranked teams** (104 teams total; the two official baseline
rows are unranked references). The organizers ran each team's final version 3 times on
each hidden card (E–H) and averaged; results were published on
2026-10-11 on the [final results page](https://create.gosim.org/survey26/platform/leaderboard/final-hidden):

| Rank | Team | **Total (mean of E–H)** | E | F | G | H |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Pleiades | 33,575.79 | 37,542.8 | 31,005.2 | 31,412.0 | 34,343.3 |
| 2 | starwing | 33,051.03 | 37,193.9 | 30,722.2 | 30,307.0 | 33,981.0 |
| 3 | 迷星叫 | 32,598.75 | 37,312.4 | 30,206.4 | 29,441.2 | 33,435.0 |
| **30** | **Aurora-X** | **27,638.79** | **33,522.55** | **24,797.69** | **21,425.09** | **30,809.83** |
| — | 基线 · 官方示例（进阶版 Python) | 27,100.86 | 33,077.9 | 25,346.0 | 20,296.6 | 29,683.0 |
| — | 基线 · 官方示例（基础版） | 24,133.52 | 30,407.7 | 22,015.0 | 23,609.1 | 20,502.2 |

Our per-card scores are the means of 3 runs each (all 12 runs finished normally). The
three evaluation means were 27,659.72 / 27,391.70 / 27,864.95 — card F alone swung
23,319–25,720 between runs, which is the night-plan run-to-run variance, not a crash.

Online phase of the competition (public cards A–D, mean of 4 cards, final version):

| Version | A–D mean | Note |
|---|---:|---|
| Organizers' pro example (baseline) | 29,710.03 | official reference row |
| v1 = pro + `PRO_REQUIRED_BONUS=160` | 29,395.50 | run without a model |
| v3 = v1 + quiet-night skip, `MODEL_WAIT_MAX=3`, `FORECAST_DISCOUNT=1.0` | 29,811.16 | run without a model |
| **v4 (final) = v3 + `PRO_REQUEST_FLOOR=500`** | **29,928.17** | +218.1 over the pro baseline |

## What we changed

The competition's hidden "realistic" cards changed the rules of the game: each card
carries ~63 paid observation requests whose `completion_reward` is 0 (the +500
settlement per observed request target is granted separately), and they hide the real
closure schedule inside multilingual, codeword-laden free text. Against that, our final
version keeps three changes over the organizers' example:

1. **Request valuation floor** (`src/planner.rs`). The example values a request as
   `REQUEST_MULT × completion_reward / targets_remaining`; with `completion_reward = 0`
   every request is valued at zero and the engine completes about 1 of 63. The fix
   values it at `max(completion_reward, REQUEST_FLOOR)` with `REQUEST_FLOOR = 500`, so
   the engine chases requests again. This is the single highest-leverage line in the
   patch, and it is card-agnostic: it only corrects a valuation formula.
2. **Quiet-night LLM skip** (`src/main.rs`, `src/advisor.rs`). On a night with nothing
   in the forecast, the bulletin, or the current notices, the model is not asked at
   all. In an earlier experiment, blocking the night start on model answers lost slots
   in a simulation that advances ~0.5 s/slot in real time; this change makes "the LLM
   is slow or down" cost exactly zero decisions, and removes most of the night-plan
   run-to-run variance on quiet nights.
3. **Submission settings** (`observer.project.json`): the container image pinned by
   digest, `PRO_MODEL_WAIT_MAX=3` (never block a decision on the model for more than
   3 s), `PRO_FORECAST_DISCOUNT=1.0`, `PRO_REQUEST_FLOOR=500`.

What did **not** make the final: a fourth LLM stage that decodes the hidden closure
intelligence from the free-text duty log (platform versions v5–v9). It had to survive a
rate-limited relay (429 storms under 8-card concurrency), endpoint reply truncation, and
a strict wall-clock budget; every attempt either died in infrastructure or landed within
the night-plan noise band, so the final stayed with the minimal, proven v4. The
experiments are not part of this repository.

## How it works

**Protocol loop and pacing (`src/main.rs`).** The binary speaks the platform's
JSON-lines protocol on stdin/stdout. The platform charges CPU time inside the agent's
turns (waiting for the model is free), so the agent paces itself with `getrusage`: an
EMA of the CPU cost per decision at each search level, scaled by `PACE_SAFETY`
(default 0.75), against the `remaining_seconds` the engine reports.

**Planner (`src/planner.rs`).** A port of the organizers' `python-pro` planner. Every
decision runs one search over pointings, with fibre-to-target assignment, exposure
duration and program selection, maximising science value under the night's constraints.
All constants are `PRO_*` environment knobs (see `Params::load`); the submission set
only the three listed above.

**LLM advisor (`src/advisor.rs`).** Three model-driven stages, all optional at run time
and all falling back to the rules on any failure:

| Stage | Where (file : function) | Trigger | What the model's output decides | Validated / clamped | Fallback |
|---|---|---|---|---|---|
| **night_plan** (NLU + task planning + plan adaptation) | `advisor.rs : Advisor::start_night` | start of every non-quiet night | `bad_night` (faint required targets wait) and `avoid_directions` from the forecast and bulletin | `avoid_directions` intersected with sectors the forecast/bulletin actually announced | no bad night, no avoided sectors |
| **fault_review** (data parsing + action decision) | `advisor.rs : Advisor::start_night / poll` | start of every non-quiet night | `fault_likely ∈ [0,1]` from the agent's own per-hour quality table, setting how readily the agent probes a fault tonight | clamped to [0,1]; invalid answers dropped | rule threshold |
| **confirm_report** (action decision) | `advisor.rs : Advisor::confirm_report` | before each paid fault report | go / no-go on a report that would cost 150 if wrong | boolean only | the rule's decision |

Calls run in the background; the agent waits only up to `MODEL_WAIT_MAX` seconds (3 s
in the submission) and keeps planning otherwise.

**Robust client (`src/llm_client.rs`).** Any OpenAI-compatible `/chat/completions`
endpoint. Call budget 1500 per run (0 when `OBSERVER_MODEL_DISABLED=1`); every failed
attempt is retried with `Retry-After` honoured on 429/5xx; bounded timeouts; no
`temperature` parameter (the Kimi Coding Plan endpoint rejects it). A `.env` file is
loaded for local runs; the platform injects its own credentials at run time.

## Build and run

```bash
cargo build --release --locked     # -> target/release/rust-pro
```

The binary speaks the hackathon JSON-lines protocol on stdin/stdout. For a local run,
pair it with the organizers' runner from the official challenge materials —
`rust_pro_entry.py` is the shim we used so a runner that launches the agent with
`python -B <entry>` starts the binary instead (it `execv`s `target/release/rust-pro`
with stdio passed through).

### Configuration

| Variable | Meaning |
|---|---|
| `OPENAI_API_KEY` (or `KIMI_API_KEY`) | model key; without it the agent runs rules only |
| `OPENAI_BASE_URL` | any OpenAI-compatible endpoint; locally e.g. the Kimi Coding Plan (`https://api.kimi.com/coding/v1`) |
| `OPENAI_MODEL` | model name (Kimi Coding Plan: `k3`) |
| `OBSERVER_MODEL_DISABLED=1` | force rules-only mode |
| `PRO_REQUEST_FLOOR=500` | **submission setting**: floor for valuing a paid observation request when `completion_reward` is 0 |
| `PRO_MODEL_WAIT_MAX=3` | **submission setting**: max seconds to wait for the night-start model answers |
| `PRO_FORECAST_DISCOUNT=1.0` | **submission setting**: trust the forecast at full value |
| `PRO_<NAME>` | any planner/advisor constant (see `Params::load` in `src/planner.rs` and `src/main.rs`) |

## Reproduce the submission

```bash
git clone <repository URL> aurora-x-pro && cd aurora-x-pro
cargo build --release --locked
```

This tree is the source of platform revision `15dfec80-0d76-49ca-acfa-df35c8e24c99`
(`Aurora-X pro-v4-reqfloor`), chosen as the team's final version on 2026-10-07. The
organizers evaluated it 3 times on each hidden card (E–H) on 2026-10-08 and published
the ranking on 2026-10-11. Dependencies are pinned by `Cargo.lock` and the evaluation
container image by digest in `observer.project.json`, so a build from this tree makes
the same decisions the ranked submission did.

## Scope of this repository

The submitted version only. Not included: the organizers' runner, task cards and docs
(they are the organizers' materials — fetch the official challenge materials instead),
our local test-card generators and evaluation harness, experiment notes, and the
intel-decoding experiments (v5–v9) described above. Our practice-phase Python agent is a
separate project: [github.com/kkkkikun/aurora-x-obp](https://github.com/kkkkikun/aurora-x-obp).

## License

The team's changes are licensed under the [Apache License 2.0](LICENSE). The project is
a derived work of the organizers' `rust-pro` example, which is licensed
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); the portions derived
from it remain under that license. See [NOTICE](NOTICE) for the per-file split.

The task cards, simulated data and evaluation code are the organizers' materials. Please
cite the GOSIM 2026 Agentic Observer Hackathon (https://create.gosim.org/survey26/).
