//! Agent Observer v4 reference agent ("rust-pro"), participant-agent-protocol-v4. A Rust port of
//! `examples/python-pro` (same strategy, same constants, same model stages).
//!
//! One JSON object per line on stdin, one per line on stdout, logs on stderr.
//!
//! What it does (details in README.md and planner.rs):
//!
//! 1. Planning: every decision picks pointing, fibre assignment, duration and program together, maximising
//!    expected gain minus a price for telescope time (gain - lambda * T). Required targets and observation
//!    requests enter as probability-weighted bonuses.
//! 2. Program: the band level is fitted to saturated hits, which show the program multiplier exactly.
//! 3. Instrument faults: E = (quality level) / (band level). Weather moves both, a fault only the first;
//!    when E stays low the agent reports. Free false reports are spent early; each low episode is probed
//!    once; paid probes need two low nights.
//! 4. Pace: the search level adapts to the measured cost per decision so a 4-month card fits the clock.
//! 5. Model (advisor.rs): at every night start a night plan (forecast + bulletin -> bad night, sectors to avoid)
//!    and a fault review (own quality table -> how likely a fault is, which gates paid reports); before a paid
//!    report the model confirms or vetoes. Calls run in the background; without an API key the agent exits.

mod advisor;
mod llm_client;
mod planner;
mod skymath;

use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, Write};
use std::time::Instant;

use advisor::{Advisor, FaultReview, NightPlan};
use llm_client::{api_key, load_dotenv, model_disabled, LlmClient};
use planner::{env_f, env_i, Planner};
use skymath::{format_date, format_hour_stamp, format_utc, parse_utc, psum, round_to};

const BAD_KINDS: [&str; 4] = ["rain", "storm", "overcast", "haze"];
const PROTOCOL: &str = "participant-agent-protocol-v4";
const MAX_FALSE_REPORTS: i64 = 8;
const PAID_SPACING_HOURS: f64 = 20.0;
const MIN_REPORT_SPACING_HOURS: f64 = 2.0;

pub fn log(text: &str) {
    let mut err = std::io::stderr().lock();
    let _ = writeln!(err, "{text}");
    let _ = err.flush();
}

/// User + system CPU time of this process (all threads), like Python's `time.process_time()`.
#[cfg(unix)]
fn process_time() -> f64 {
    // SAFETY: getrusage only writes into the zeroed struct we hand it.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return 0.0;
    }
    let seconds = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1e6;
    seconds(usage.ru_utime) + seconds(usage.ru_stime)
}

#[cfg(not(unix))]
fn process_time() -> f64 {
    0.0
}

/// Tunable constants of the agent (each one overridable with `PRO_<NAME>`).
struct Knobs {
    // --- fault reporting (see fault_verdict) ---
    e_low_free: f64,     // free probe: E below this in 3 of the last 4 hours
    e_low_free2: f64,    // ... threshold while both free probes are left
    e_free_hours: usize, //
    e_low: f64,          // paid probe: low hours must span two nights ...
    e_paid_low: f64,     // ... the median of the last 12 hourly E below this ...
    e_paid_hours: usize, //
    e_recover: f64,      // after a false probe, wait until E is back above this
    max_paid_false: i64, //
    persist_nights: usize,
    e_paid_step: f64, // ... minus this per paid false probe so far
    // The participant guide: an earthquake (announced in the bulletin) lowers instrument efficiency, the loss
    // fades night by night, and a report does not repair it. So E drops right after an earthquake are not
    // reportable, and while its effect may last only a new step down in E (a fresh drop from the preceding
    // hours) is fault evidence.
    quake_hold_hours: f64, // no probes this long after an earthquake notice appears
    quake_step: f64,       // step: median E of the last 3 rows < this x the 9 rows before
    quake_tail_hours: f64, // the earthquake period lasts this long after its last notice
    // --- pace ---
    pace_safety: f64,
    // --- model ---
    model_wait_max: f64,   // longest wait for the night's model answers (s)
    model_fault_high: f64, // fault review at or above this: report more readily tonight
    model_fault_low: f64,  // ... at or below this: paid reports need the strongest evidence
    scale_step: f64,
    model_free_probe: bool, // 1: a high fault review may also spend a free probe on a low scale
    fixed_level: i64,       // development only: pin the search level (deterministic runs)
}

impl Knobs {
    fn load() -> Knobs {
        Knobs {
            e_low_free: env_f("E_LOW_FREE", 0.9),
            e_low_free2: env_f("E_LOW_FREE2", 0.9),
            e_free_hours: env_i("E_FREE_HOURS", 4).max(1) as usize,
            e_low: env_f("E_LOW", 0.85),
            e_paid_low: env_f("E_PAID_LOW", 0.75),
            e_paid_hours: env_i("E_PAID_HOURS", 12).max(1) as usize,
            e_recover: env_f("E_RECOVER", 0.95),
            max_paid_false: env_i("MAX_PAID", 6),
            persist_nights: env_i("PERSIST_NIGHTS", 3).max(1) as usize,
            e_paid_step: env_f("E_PAID_STEP", 0.05),
            quake_hold_hours: env_f("QUAKE_HOLD_HOURS", 12.0),
            quake_step: env_f("QUAKE_STEP", 0.8),
            quake_tail_hours: env_f("QUAKE_TAIL_HOURS", 24.0),
            pace_safety: env_f("PACE_SAFETY", 0.75),
            model_wait_max: env_f("MODEL_WAIT_MAX", 20.0),
            model_fault_high: env_f("MODEL_FAULT_HIGH", 0.6),
            model_fault_low: env_f("MODEL_FAULT_LOW", 0.15),
            scale_step: env_f("SCALE_STEP", 0.7),
            model_free_probe: env_i("MODEL_FREE_PROBE", 0) != 0,
            fixed_level: env_i("FIXED_LEVEL", -1),
        }
    }
}

fn median_of(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

fn action_wait_until(until: f64, reason: &str) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("action".into(), json!("wait"));
    m.insert("until_utc".into(), json!(format_utc(until)));
    m.insert("reason".into(), json!(reason));
    m
}

fn action_wait_for(seconds: i64, reason: &str) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("action".into(), json!("wait"));
    m.insert("duration_seconds".into(), json!(seconds));
    m.insert("reason".into(), json!(reason));
    m
}

fn action_finish(reason: &str) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("action".into(), json!("finish"));
    m.insert("reason".into(), json!(reason));
    m
}

struct ObserverAgent {
    k: Knobs,
    planner: Planner,
    client: LlmClient,
    advisor: Advisor,
    model_wait: f64,          // wall seconds spent waiting for the model (not planning cost)
    fault_likely: Option<f64>, // tonight's model estimate that an instrument fault is active
    scale_hours: BTreeMap<i64, Vec<f64>>, // hour -> [planner.scale samples] (for the model's fault table)
    start: f64,
    forecast_notices: Vec<Value>,
    night_seen: Option<usize>,
    observes: i64,
    // fault reporting state
    free_allowance: i64,
    reports: i64,
    correct_reports: i64,
    false_reports: i64,
    false_since_correct: i64,
    paid_false: i64,
    last_report_hours: f64,
    ref_from_hours: f64,
    episode_blocked: bool,
    blocked_at_hour: i64,
    quake_on: bool,
    quake_onset_hours: f64,
    quake_last_hours: f64,
    logged_hour: Option<i64>,
    // pace state
    cost_ema: [f64; 4], // CPU seconds per observe decision at each search level
    wall_ema: [f64; 4], // real seconds per observe decision (own turn, model waits excluded)
    turn_end: Option<Instant>,
    decisions: i64,
    engine_ema: Option<f64>,
    sim_step_ema: Option<f64>,
    last_now: Option<f64>,
}

impl ObserverAgent {
    fn new(init: &Value) -> ObserverAgent {
        let started = Instant::now();
        let planner = Planner::new(init);
        let client = LlmClient::new();
        let free_allowance = init["scoring"]["reporting"]["false_report_free_allowance"].as_f64().unwrap_or(0.0) as i64;
        let agent = ObserverAgent {
            k: Knobs::load(),
            start: parse_utc(init["survey"]["start_utc"].as_str().unwrap_or("")),
            advisor: Advisor::new(),
            model_wait: 0.0,
            fault_likely: None,
            scale_hours: BTreeMap::new(),
            forecast_notices: Vec::new(),
            night_seen: None,
            observes: 0,
            free_allowance,
            reports: 0,
            correct_reports: 0,
            false_reports: 0,
            false_since_correct: 0,
            paid_false: 0,
            last_report_hours: -1e9,
            ref_from_hours: -1e9,
            episode_blocked: false,
            blocked_at_hour: -1,
            quake_on: false,
            quake_onset_hours: -1e9,
            quake_last_hours: -1e9,
            logged_hour: None,
            cost_ema: [0.0; 4],
            wall_ema: [0.0; 4],
            turn_end: None,
            decisions: 0,
            engine_ema: None,
            sim_step_ema: None,
            last_now: None,
            planner,
            client,
        };
        log(&format!(
            "pro: {} targets, {} required, {} nights; init {:.2}s; model {}",
            agent.planner.ids.len(),
            agent.planner.required.iter().filter(|&&r| r).count(),
            agent.planner.nights.len(),
            started.elapsed().as_secs_f64(),
            agent.client.model
        ));
        agent
    }

    // --- decision loop ------------------------------------------------------------------------------

    fn respond(&mut self, payload: &Value) -> Map<String, Value> {
        let started = Instant::now();
        let cpu_started = process_time();
        if let Some(end) = self.turn_end {
            // engine time between our turns (charged only by the old real-time clock)
            let gap = started.duration_since(end).as_secs_f64();
            if (0.0..5.0).contains(&gap) {
                self.engine_ema = Some(match self.engine_ema {
                    None => gap,
                    Some(e) => 0.95 * e + 0.05 * gap,
                });
            }
        }
        let model_before = self.model_wait;
        let level = self.planner.fast_level;
        let action = self.respond_inner(payload);
        // the platform charges CPU time inside our turns; waiting for the model is free of CPU, so keep it
        // out of the real-time estimate as well
        let cpu = process_time() - cpu_started;
        let wall = started.elapsed().as_secs_f64() - (self.model_wait - model_before);
        if action.get("action").and_then(Value::as_str) == Some("observe") && level < 4 {
            for (ema, cost) in [(&mut self.cost_ema, cpu), (&mut self.wall_ema, wall)] {
                let c = ema[level];
                ema[level] = if c == 0.0 { cost } else { 0.9 * c + 0.1 * cost };
                for k in level + 1..4 {
                    // cheaper levels not measured yet: a third of the level above
                    if ema[k] == 0.0 || ema[k] > ema[k - 1] {
                        ema[k] = ema[k - 1] / 3.0;
                    }
                }
            }
        }
        self.turn_end = Some(Instant::now());
        action
    }

    fn respond_inner(&mut self, payload: &Value) -> Map<String, Value> {
        let now_utc = payload["now_utc"].as_str().unwrap_or("").to_string();
        let now = parse_utc(&now_utc);
        if let Some(last_now) = self.last_now {
            if self.planner.current_night(now).is_some() {
                let step = now - last_now;
                if 0.0 < step && step <= 3600.0 {
                    self.sim_step_ema = Some(match self.sim_step_ema {
                        None => step,
                        Some(e) => 0.95 * e + 0.05 * step,
                    });
                }
            }
        }
        self.last_now = Some(now);
        let hours = (now - self.start) / 3600.0;
        let empty = Vec::new();
        let messages = payload.get("new_messages").and_then(Value::as_array).unwrap_or(&empty);
        for message in messages {
            if message.get("record_type").and_then(Value::as_str) == Some("forecast") {
                self.forecast_notices = message.get("notices").and_then(Value::as_array).cloned().unwrap_or_default();
            }
        }
        let last = payload.get("last_result").cloned().unwrap_or(Value::Null);
        if last.get("action").and_then(Value::as_str) == Some("report") {
            self.on_report_result(&last, hours);
        }
        let bulletin = payload.get("latest_bulletin").cloned().unwrap_or(Value::Null);
        self.planner.on_messages(messages, &bulletin);
        let quake = self.planner.notices.iter().any(|(kind, _)| kind == "earthquake");
        if quake && !self.quake_on {
            self.quake_onset_hours = hours;
            log(&format!("pro: earthquake notice at {now_utc}"));
        }
        self.quake_on = quake;
        if quake {
            self.quake_last_hours = hours;
        }
        let requests = payload.get("active_requests").and_then(Value::as_array).unwrap_or(&empty);
        self.planner.on_requests(requests);
        self.planner.on_result(&last, hours);
        self.pace(payload, now);

        let Some((night_index, night_start, night_end)) = self.planner.current_night(now) else {
            return match self.planner.next_night_start(now) {
                None => action_finish("no observing night left"),
                Some(nxt) => action_wait_until(nxt, "daytime: sleep until the next night"),
            };
        };
        if self.night_seen != Some(night_index) {
            self.night_seen = Some(night_index);
            self.night_advice(night_start, payload, hours);
        } else {
            let (plan, fault) = self.advisor.poll(&mut self.client);
            self.apply_advice(plan, fault);
        }
        let scale = self.planner.scale;
        self.scale_hours.entry(hours.trunc() as i64).or_default().push(scale);
        if night_end - now < self.planner.min_exposure {
            return match self.planner.next_night_start(now) {
                None => action_finish("survey over"),
                Some(nxt) => action_wait_until(nxt, "night ending"),
            };
        }
        if self.planner.site_closed() {
            return action_wait_for(self.to_next_slot(now, night_start), "bulletin: rain/storm over the whole sky");
        }
        if let Some(report) = self.maybe_report(hours, payload, &now_utc) {
            return report;
        }
        match self.planner.plan(now, night_end, night_index, hours) {
            None => action_wait_for(self.to_next_slot(now, night_start), "nothing useful is up"),
            Some(mut action) => {
                self.observes += 1;
                let fibres = action.get("assignments").and_then(Value::as_object).map_or(0, |a| a.len());
                let program = action.get("program").and_then(Value::as_str).unwrap_or("").to_string();
                action.insert("reason".into(), json!(format!("{fibres} fibres, program {program}")));
                action
            }
        }
    }

    fn to_next_slot(&self, now: f64, night_start: f64) -> i64 {
        let slot = self.planner.slot_seconds as f64;
        let into = skymath::pmod(now - night_start, slot);
        (slot - into).min(3600.0).max(60.0) as i64
    }

    // --- pace ---------------------------------------------------------------------------------------

    /// (CPU seconds left, real seconds left, fair clock?) from the request's wallclock block.
    ///
    /// Fair clock (current platform): the budget is normalized CPU time inside our turns, and
    /// remaining_real_cpu_seconds converts it to this machine's CPU seconds; a separate real-time cap
    /// (wall_remaining_seconds) only guards against runaway runs. Older runners count real time only.
    fn clock(payload: &Value) -> (f64, f64, bool) {
        let wall = &payload["wallclock"];
        if let Some(cpu) = wall.get("remaining_real_cpu_seconds").filter(|v| !v.is_null()) {
            let wall_left = wall.get("wall_remaining_seconds").map(planner::num).unwrap_or(1e9);
            return (planner::num(cpu), wall_left, true);
        }
        let remaining = wall.get("remaining_seconds").map(planner::num).unwrap_or(1e9);
        (remaining, remaining, false)
    }

    fn decisions_left(&self, now: f64) -> f64 {
        let night_seconds = psum(self.planner.nights.iter().filter(|&&(_, e)| e > now).map(|&(s, e)| (e - s.max(now)).max(0.0)));
        (night_seconds / self.sim_step_ema.unwrap_or(900.0)).max(1.0) // daytime waits cost nothing
    }

    /// Pick the search level from the measured cost per decision and the decisions still to come.
    fn pace(&mut self, payload: &Value, now: f64) {
        let (cpu_left, wall_left, fair) = Self::clock(payload);
        let decisions_left = self.decisions_left(now);
        let engine = self.engine_ema.unwrap_or(0.0);
        let cpu_budget = if fair { self.k.pace_safety * cpu_left / decisions_left } else { 1e9 };
        let wall_budget = self.k.pace_safety * wall_left / decisions_left - engine;
        // estimates of levels not used for a while decay, so the agent climbs back up and re-measures them
        self.decisions += 1;
        if self.decisions % 50 == 0 {
            for k in 0..4 {
                if k != self.planner.fast_level {
                    self.cost_ema[k] *= 0.85;
                    self.wall_ema[k] *= 0.85;
                }
            }
        }
        if self.k.fixed_level >= 0 {
            self.planner.fast_level = self.k.fixed_level as usize;
            return;
        }
        let mut level = 0;
        while level < 3 && (self.cost_ema[level] > cpu_budget || self.wall_ema[level] > wall_budget) {
            level += 1;
        }
        if cpu_left.min(wall_left) < 15.0 {
            level = 4;
        }
        if level != self.planner.fast_level {
            log(&format!(
                "pro: pace level {} (cpu budget {:.0} ms, wall budget {:.0} ms, cpu costs {:?} ms, {:.0} decisions left)",
                level,
                cpu_budget.min(99.0) * 1000.0,
                wall_budget * 1000.0,
                self.cost_ema.iter().map(|c| (c * 1000.0).round() as i64).collect::<Vec<_>>(),
                decisions_left
            ));
            self.planner.fast_level = level;
        }
    }

    // --- model stages (advisor.rs): night plan and fault review at every night start --------------------

    /// Night start: rule defaults first, then the two model calls (night plan, fault review).
    fn night_advice(&mut self, night_start: f64, payload: &Value, hours: f64) {
        let night_date = format_date(night_start - 12.0 * 3600.0);
        let tonight: Vec<Value> = self
            .forecast_notices
            .iter()
            .filter(|n| n.get("nights").and_then(Value::as_array).map_or(false, |a| a.iter().any(|d| d.as_str() == Some(night_date.as_str()))))
            .cloned()
            .collect();
        let bulletin: Vec<Value> = payload["latest_bulletin"].get("notices").and_then(Value::as_array).cloned().unwrap_or_default();
        // rule defaults, kept when the model gives no valid answer
        self.planner.bad_forecast = tonight.iter().any(|n| {
            n.get("direction").and_then(Value::as_str) == Some("ALL")
                && n.get("event_kind").and_then(Value::as_str).map_or(false, |k| BAD_KINDS.contains(&k))
        });
        self.planner.extra_avoid = BTreeSet::new();
        self.fault_likely = None;
        let left = Self::clock(payload).1;
        let started = Instant::now();
        let table = self.fault_table(hours);
        // quiet night (nothing in the forecast, the bulletin, or the current notices): don't ask the model
        let ask = !(tonight.is_empty() && bulletin.is_empty() && self.planner.notices.is_empty());
        let wait = if ask { self.model_wait_budget(payload) } else { 0.0 };
        let (plan, fault) = self.advisor.start_night(&mut self.client, &night_date, &tonight, &bulletin, table, left, wait, ask);
        self.model_wait += started.elapsed().as_secs_f64();
        self.apply_advice(plan, fault);
    }

    /// How long a night start may wait for the model. Waiting costs no CPU budget, only real time: use half
    /// of the real time the planner and the engine will not need, spread over the nights left.
    fn model_wait_budget(&self, payload: &Value) -> f64 {
        let (_, wall_left, _) = Self::clock(payload);
        let now = self.last_now.unwrap_or(0.0);
        let nights_left = self.planner.nights.iter().filter(|&&(_, e)| e > now).count().max(1) as f64;
        let per_decision = self.wall_ema[self.planner.fast_level.min(3)].max(0.05) + self.engine_ema.unwrap_or(0.02);
        let spare = wall_left - 1.5 * self.decisions_left(now) * per_decision - 60.0;
        (0.5 * spare / nights_left).min(self.k.model_wait_max).max(0.0)
    }

    fn apply_advice(&mut self, plan: Option<NightPlan>, fault: Option<FaultReview>) {
        if let Some(plan) = plan {
            self.planner.bad_forecast = plan.bad_night;
            self.planner.extra_avoid = plan.avoid_directions.iter().cloned().collect();
            log(&format!(
                "llm night plan {}: bad_night={} avoid={:?} ({})",
                self.advisor.night_date, plan.bad_night, plan.avoid_directions, plan.reason
            ));
        }
        if let Some(fault) = fault {
            self.fault_likely = Some(fault.fault_likely);
            log(&format!("llm fault review {}: fault_likely={:.2} ({})", self.advisor.night_date, fault.fault_likely, fault.reason));
        }
    }

    /// Usual clear-sky scale since the last repair: 75th percentile of the hourly medians.
    fn scale_ref(&self) -> f64 {
        let mut values: Vec<f64> = self
            .scale_hours
            .iter()
            .filter(|(h, v)| **h as f64 >= self.ref_from_hours && !v.is_empty())
            .map(|(_, v)| median_of(v))
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if values.len() >= 4 {
            values[(3 * values.len()) / 4]
        } else {
            1.0
        }
    }

    /// The evidence the fault review reads: the last ~30 observed hours.
    fn fault_table(&self, hours: f64) -> Value {
        let e_by_hour: BTreeMap<i64, f64> = self
            .planner
            .e_hours
            .iter()
            .filter(|(h, _, _)| *h as f64 >= self.ref_from_hours)
            .map(|(h, _, v)| (*h, median_of(v)))
            .collect();
        let keys: Vec<i64> = self.scale_hours.keys().cloned().collect();
        let mut rows = Vec::new();
        for &hour in &keys[keys.len().saturating_sub(30)..] {
            if (hour as f64) < self.ref_from_hours {
                continue;
            }
            let stamp = format_hour_stamp(self.start + hour as f64 * 3600.0);
            let e = e_by_hour.get(&hour).map_or(Value::Null, |e| json!(round_to(*e, 2)));
            rows.push(json!([stamp, e, round_to(median_of(&self.scale_hours[&hour]), 2)]));
        }
        let notices: BTreeSet<String> = self.planner.notices.iter().map(|(k, d)| format!("{k} {d}")).collect();
        json!({
            "columns": ["utc_hour", "E", "scale"],
            "rows": rows,
            "ref": round_to(self.scale_ref(), 2),
            "notices_now": notices.into_iter().collect::<Vec<_>>(),
            "hours_since_earthquake_notice_began":
                if self.quake_onset_hours < -1e8 { Value::Null } else { json!(round_to(hours - self.quake_onset_hours, 1)) },
            "free_false_reports_left": self.free_left().max(0),
            "paid_false_reports_so_far": self.paid_false,
            "correct_reports_so_far": self.correct_reports,
            "hours_since_last_report": if self.reports == 0 { Value::Null } else { json!(round_to(hours - self.last_report_hours, 1)) },
        })
    }

    // --- instrument faults ---------------------------------------------------------------------------

    /// Report (probe) when the quality level stays below what the program bands allow.
    ///
    /// A report costs no time, its answer arrives at once, and the first false reports after each correct
    /// one are free: spend free probes readily, paid ones only on strong, lasting evidence.
    fn maybe_report(&mut self, hours: f64, payload: &Value, now_utc: &str) -> Option<Map<String, Value>> {
        if self.false_reports >= MAX_FALSE_REPORTS || hours - self.last_report_hours < MIN_REPORT_SPACING_HOURS {
            return None;
        }
        if hours - self.quake_onset_hours < self.k.quake_hold_hours {
            return None; // the earthquake explains the drop; a report would not repair it
        }
        if self.fault_verdict(hours, now_utc) && self.model_agrees(hours, payload, now_utc) {
            self.last_report_hours = hours;
            self.reports += 1;
            log(&format!("pro: report at {now_utc} (quality below what the program bands allow), free left {}", self.free_left()));
            let mut m = Map::new();
            m.insert("action".into(), json!("report"));
            m.insert("reason".into(), json!("quality level below what the program bands allow"));
            return Some(m);
        }
        None
    }

    /// Hourly E = quality level / band level (planner.e_hours). 1 = consistent; a fault keeps E low.
    fn fault_verdict(&mut self, hours: f64, now_utc: &str) -> bool {
        let k = &self.k;
        let rows: Vec<(i64, usize, f64)> = self
            .planner
            .e_hours
            .iter()
            .filter(|(h, _, _)| *h as f64 >= self.ref_from_hours)
            .map(|(h, n, v)| (*h, *n, median_of(v)))
            .collect();
        let hour_now = hours.trunc() as i64;
        if !rows.is_empty() && self.logged_hour != Some(hour_now) {
            self.logged_hour = Some(hour_now);
            log(&format!(
                "pro: E {} {:.2} scale {:.3} band {:.3}",
                now_utc,
                rows[rows.len() - 1].2,
                self.planner.scale,
                self.planner.band_level.unwrap_or(0.0)
            ));
        }
        if rows.len() < k.e_free_hours || rows[rows.len() - 1].0 < hour_now - 1 {
            return false;
        }
        if k.quake_step > 0.0 && hours - self.quake_last_hours < k.quake_tail_hours {
            if rows.len() < 8 {
                return false;
            }
            let n = rows.len();
            let last = median_of(&rows[n - 3..].iter().map(|r| r.2).collect::<Vec<_>>());
            let prev = median_of(&rows[n.saturating_sub(12)..n - 3].iter().map(|r| r.2).collect::<Vec<_>>());
            if !(last < k.quake_step * prev && rows[n - 3].0 >= hour_now - 4) {
                return false;
            }
            if self.episode_blocked && rows[n - 3].0 <= self.blocked_at_hour {
                return false; // the step that was already probed, not a new one
            }
            self.episode_blocked = false; // a new step is a new episode
        }
        if self.episode_blocked {
            // this low episode was probed already and was not a fault: wait for a recovery first
            let last4 = &rows[rows.len().saturating_sub(4)..];
            if last4.iter().filter(|r| r.2 >= k.e_recover).count() >= 3 && rows[rows.len() - 1].0 > self.blocked_at_hour {
                self.episode_blocked = false;
                log(&format!("pro: quality recovered at {now_utc}; probing re-armed"));
            } else {
                return false;
            }
        }
        let likely = self.fault_likely;
        if k.model_free_probe && likely.map_or(false, |l| l >= k.model_fault_high) && self.free_left() > 0 && self.scale_step_low() {
            // off by default: on the practice cards it spent free probes on unannounced weather
            log(&format!("pro: model-flagged fault (likely {:.2}) and scale below {} x ref", likely.unwrap(), k.scale_step));
            return true;
        }
        if self.free_left() > 0 {
            let last = &rows[rows.len().saturating_sub(k.e_free_hours)..];
            let threshold = if self.free_left() >= 2 { k.e_low_free2 } else { k.e_low_free };
            let low = last.iter().filter(|r| r.2 < threshold).count();
            return low + 1 >= k.e_free_hours
                && last[last.len() - 1].2 < threshold
                && last[last.len() - 1].0 - last[0].0 <= k.e_free_hours as i64 + 3;
        }
        if self.paid_false >= k.max_paid_false || hours - self.last_report_hours < PAID_SPACING_HOURS {
            return false;
        }
        let last = &rows[rows.len().saturating_sub(k.e_paid_hours)..];
        let values: Vec<f64> = last.iter().map(|r| r.2).collect();
        let nights: BTreeSet<usize> = last.iter().filter(|r| r.2 < k.e_low).map(|r| r.1).collect();
        // a fault never goes away on its own: three low nights in a row justify a probe whatever the bar
        let mut by_night: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
        for r in &rows {
            by_night.entry(r.1).or_default().push(r.2);
        }
        let night_keys: Vec<usize> = by_night.keys().cloned().collect();
        let nights_seq = &night_keys[night_keys.len().saturating_sub(k.persist_nights)..];
        if nights_seq.len() == k.persist_nights
            && nights_seq[nights_seq.len() - 1] - nights_seq[0] <= k.persist_nights
            && nights_seq.iter().all(|n| by_night[n].len() >= 3 && median_of(&by_night[n]) < k.e_low)
            && hours - self.last_report_hours >= 40.0
        {
            return true;
        }
        if likely.map_or(false, |l| l <= k.model_fault_low) {
            return false; // the fault review sees weather, not a fault: only the persistence rule above may report
        }
        // each paid false probe raises the bar for the next one
        let paid_low = k.e_paid_low - k.e_paid_step * self.paid_false as f64;
        last.len() == k.e_paid_hours
            && median_of(&values) < paid_low
            && nights.len() >= 2
            && last[last.len().saturating_sub(3)..].iter().all(|r| r.2 < k.e_low)
    }

    /// The last 3 observed hours all sit below SCALE_STEP x the usual clear-sky scale.
    fn scale_step_low(&self) -> bool {
        let recent: Vec<f64> = self
            .scale_hours
            .iter()
            .filter(|(h, v)| **h as f64 >= self.ref_from_hours && !v.is_empty())
            .map(|(_, v)| median_of(v))
            .collect();
        let recent = &recent[recent.len().saturating_sub(3)..];
        recent.len() == 3 && recent.iter().cloned().fold(f64::MIN, f64::max) < self.k.scale_step * self.scale_ref()
    }

    /// Paid probes only: the model looks at the evidence first and may veto. Free probes cost nothing, so they
    /// never wait for it. No answer in time: the rule's decision stands.
    fn model_agrees(&mut self, hours: f64, payload: &Value, now_utc: &str) -> bool {
        if self.free_left() > 0 {
            return true;
        }
        let rows: Vec<f64> = self
            .planner
            .e_hours
            .iter()
            .filter(|(h, _, _)| *h as f64 >= self.ref_from_hours)
            .map(|(_, _, v)| median_of(v))
            .collect();
        let rows = &rows[rows.len().saturating_sub(24)..];
        let evidence = json!({
            "hourly_E_last_24h": rows.iter().map(|e| round_to(*e, 2)).collect::<Vec<_>>(),
            "fault_table": self.fault_table(hours),
            "paid_false_reports_so_far": self.paid_false,
            "correct_reports_so_far": self.correct_reports,
        });
        let started = Instant::now();
        let left = Self::clock(payload).1;
        let wait = 30.0f64.min(2.0 * self.model_wait_budget(payload));
        let verdict = self.advisor.confirm_report(&mut self.client, evidence, left, wait);
        self.model_wait += started.elapsed().as_secs_f64();
        if verdict == Some(false) {
            log(&format!("pro: model vetoed a paid report at {now_utc}"));
            self.last_report_hours = hours;
            return false;
        }
        true
    }

    fn free_left(&self) -> i64 {
        self.free_allowance - self.false_since_correct
    }

    fn on_report_result(&mut self, result: &Value, hours: f64) {
        let delta = result.get("score_delta").cloned().unwrap_or(Value::Null);
        if result.get("correct").and_then(Value::as_bool).unwrap_or(false) {
            log(&format!("pro: report correct, fault repaired (delta {delta})"));
            self.correct_reports += 1;
            self.false_since_correct = 0;
            self.planner.forget_quality_history();
            self.ref_from_hours = hours;
        } else {
            self.false_since_correct += 1;
            self.false_reports += 1;
            self.episode_blocked = true;
            self.blocked_at_hour = hours.trunc() as i64;
            if self.false_since_correct > self.free_allowance {
                self.paid_false += 1;
            }
            log(&format!("pro: report false (delta {delta}); free left {}", self.free_left()));
        }
    }
}

fn main() {
    load_dotenv(std::path::Path::new(".env"));
    let rules_only = model_disabled();
    if rules_only {
        log("pro: OBSERVER_MODEL_DISABLED=1, running rules only (no model calls)");
    } else if api_key().is_empty() {
        log("missing API key: set OPENAI_API_KEY");
        std::process::exit(2);
    }
    // a panic inside a decision is caught and answered with a wait; keep its message short
    std::panic::set_hook(Box::new(|info| log(&format!("pro: panic {info}"))));
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    let mut agent: Option<ObserverAgent> = None;
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let message: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                log(&format!("pro: unreadable line ({e})"));
                continue;
            }
        };
        let kind = message.get("message_type").and_then(Value::as_str).unwrap_or("");
        if message.get("protocol_version").and_then(Value::as_str) != Some(PROTOCOL) {
            log(&format!("pro: unexpected protocol {}", message.get("protocol_version").unwrap_or(&Value::Null)));
        }
        match kind {
            "initialize" => agent = Some(ObserverAgent::new(&message["payload"])),
            "decision_request" => {
                let payload = &message["payload"];
                let result = match agent.as_mut() {
                    Some(a) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.respond(payload))).ok(),
                    None => None,
                };
                let mut action = result.unwrap_or_else(|| {
                    // never crash the run: wait one slot instead
                    log("pro: error in a decision; waiting one slot");
                    action_wait_for(900, "internal error")
                });
                action.entry("decision_source").or_insert(json!(if rules_only { "rules" } else { "llm-advised" }));
                let mut out = Map::new();
                out.insert("protocol_version".into(), json!(PROTOCOL));
                out.insert("message_type".into(), json!("decision_response"));
                out.insert("decision_sequence".into(), message.get("decision_sequence").cloned().unwrap_or(Value::Null));
                out.extend(action);
                let _ = writeln!(stdout, "{}", Value::Object(out));
                let _ = stdout.flush();
            }
            "finish" => {
                let reason = message["payload"].get("termination_reason").cloned().unwrap_or(Value::Null);
                let (observes, reports) = agent.as_ref().map_or((0, 0), |a| (a.observes, a.reports));
                log(&format!("pro finished: termination_reason={reason} observes={observes} reports={reports}"));
            }
            _ => {}
        }
    }
}
