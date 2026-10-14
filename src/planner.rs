//! Planner of the pro agent: pointing, fibre assignment, duration and program in one search.
//! Port of `python-pro/planner.py` (same constants, same order of operations).
//!
//! Per decision (plan):
//! 1. Visible targets with something left to gain are ranked with a cheap proxy; the best POOL go on.
//! 2. The gain of one exposure of duration T for target i is
//!        weight * (reach(T) * program multiplier - best so far)
//!      + REQUIRED_BONUS * P(required target reaches factor 0.5)   (discounted while its sky is far from its best)
//!      + request bonus  * P(request target reaches its threshold)
//!    times an urgency factor for targets with few nights left.
//! 3. Candidate fields: the N_ANCHORS best targets centred on each of the 16 fibres, plus the N_DENSE densest
//!    patches of remaining science. Every fibre gets the target with the largest gain there, for each
//!    duration; the winner maximises  total gain - lambda * T  (lambda = time price, scaled by the card's
//!    time scarcity), and is then refined by small pointing shifts.
//! 4. Program: the one with the largest expected score, using a band level fitted to saturated hits.
//! 5. The command is the chosen centre minus the learned pointing offset (Hard-mode cards).
//!
//! Learning from results: unsaturated hits give the quality level (scale); saturated hits show the program
//! multiplier exactly and so bracket the band level; E = quality level / band level is the instrument-fault
//! signal (clean_e); hit/miss patterns reveal the pointing offset. The planner only uses the public catalogue,
//! the public score formula, bulletins and its own results.

use serde_json::{json, Map, Value};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use crate::log;
use crate::skymath::{
    local_sidereal_deg, max_hour_angle_deg, normalized_airmass, parse_utc, pmod, psum, radec_to_altaz, round_to,
    shift_altaz, tangent_offsets, wrap180, FiberGrid, LunarModel, Moon, SIDEREAL_DEG_PER_SECOND,
};

/// `PRO_<NAME>` environment override of a float constant.
pub fn env_f(name: &str, default: f64) -> f64 {
    std::env::var(format!("PRO_{name}")).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

/// `PRO_<NAME>` environment override of an integer constant.
pub fn env_i(name: &str, default: i64) -> i64 {
    std::env::var(format!("PRO_{name}")).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

const TYPICAL_Q: f64 = 0.6;
const DENSE_FIBERS: [usize; 4] = [5, 6, 9, 10];
const DURATIONS: [f64; 11] = [300.0, 450.0, 600.0, 750.0, 900.0, 1200.0, 1500.0, 1800.0, 2400.0, 3000.0, 3600.0];
const LEVEL_DURATIONS_1: [f64; 7] = [300.0, 600.0, 900.0, 1200.0, 1800.0, 2400.0, 3600.0];
const LEVEL_DURATIONS_2: [f64; 4] = [450.0, 900.0, 1800.0, 3600.0];
const LEVEL_DURATIONS_3: [f64; 2] = [900.0, 1800.0];
const MIN_VISIBLE_SECONDS: f64 = 600.0;
const ALT_MARGIN_DEG: f64 = 0.6; // keep targets this far above the altitude limit
const REQUIRED_SAFE_FACTOR: f64 = 0.62; // a required target counts as safe at this estimated factor
const DONE_FACTOR: f64 = 0.95; // other targets are done at this factor
// --- pointing offset (Hard-mode cards: a fixed, unannounced offset; its size is not published) ---
const OFFSET_STEPS: i64 = 16; // coarse grid half-width in steps of pitch/12; widens by half on an edge hit
const OFFSET_MIN_MISSES: i64 = 6; // start estimating after this many assigned-but-missed targets
const OFFSET_REFINE_EVERY: i64 = 10; // observes between fine searches
const OFFSET_FINE_EVIDENCE: usize = 150; // observes used by the fine search
const OFFSET_MARGIN: i64 = 8; // adopt an offset only if it explains this many more outcomes
// --- learning ---
const SKY_MEMORY_HOURS: f64 = 2.0; // quality samples older than this are stale
const BAND_MEMORY_HOURS: f64 = 2.0; // saturated hits used to fit the band level
const CLOSED_KINDS: [&str; 2] = ["rain", "storm"];
const SKY_WEATHER_KINDS: [&str; 5] = ["rain", "storm", "overcast", "haze", "cold_snap"];
const BLOCKING_KINDS: [&str; 2] = ["terrain_obstruction", "rocket_launch"];

fn direction_az(direction: &str) -> Option<f64> {
    Some(match direction {
        "N" => 0.0,
        "NE" => 45.0,
        "E" => 90.0,
        "SE" => 135.0,
        "S" => 180.0,
        "SW" => 225.0,
        "W" => 270.0,
        "NW" => 315.0,
        _ => return None,
    })
}

fn az_distance(a: f64, b: f64) -> f64 {
    wrap180(a - b).abs()
}

/// Tunable constants (each one overridable with `PRO_<NAME>`).
struct Params {
    lambda_frac: f64,
    lambda_ema: f64,
    scarcity_ref: f64,
    scarcity_power: f64,
    n_anchors: usize,
    n_dense: usize,
    dense_bin_deg: f64,
    refine_rounds: i64,
    refine_fixed_t: bool,
    refine_steps: Vec<(f64, f64)>,
    pool: usize,
    neighbour_radius_deg: f64,
    edge_margin_deg: f64,
    min_t: f64,
    plan_factor_safety: f64,
    required_bonus: f64,
    req_p_lo: f64,
    req_p_hi: f64,
    request_mult: f64,
    request_floor: f64,
    req_calib_power: f64,
    forecast_discount: f64,
    req_calendar: bool,
    req_timing: f64,
    req_timing_nights: i64,
    req_timing_discount: f64,
    urgency: f64,
    partial_discount: f64,
    partial_done: f64,
    partial_nights: i64,
    plan_util: f64,
    plan_q: f64,
    kappa: f64,
    band_opt: f64,
    band_half_life: f64,
    band_fallback: bool,
    band_fallback_hours: f64,
    band_cont: bool,
    weather_gate: bool,
}

impl Params {
    fn load() -> Params {
        let refine = env_f("REFINE", 0.1);
        let mut refine_steps = Vec::new();
        if refine > 0.0 {
            for dn in [-1.0, 0.0, 1.0] {
                for de in [-1.0, 0.0, 1.0] {
                    if dn != 0.0 || de != 0.0 {
                        refine_steps.push((dn * refine, de * refine));
                    }
                }
            }
        }
        Params {
            lambda_frac: env_f("LAMBDA_FRAC", 0.6),
            lambda_ema: env_f("LAMBDA_EMA", 0.03),
            scarcity_ref: env_f("SCARCITY_REF", 0.86),
            scarcity_power: env_f("SCARCITY_POWER", 1.0),
            n_anchors: env_i("N_ANCHORS", 12).max(0) as usize,
            n_dense: env_i("N_DENSE", 20).max(0) as usize,
            dense_bin_deg: env_f("DENSE_BIN_DEG", 2.5),
            refine_rounds: env_i("REFINE_ROUNDS", 4),
            refine_fixed_t: env_i("REFINE_FIXED_T", 1) != 0,
            refine_steps,
            pool: env_i("POOL", 600).max(1) as usize,
            neighbour_radius_deg: env_f("NEIGHBOUR_RADIUS_DEG", 2.1),
            edge_margin_deg: env_f("EDGE_MARGIN_DEG", 0.04),
            min_t: env_i("MIN_T", 0) as f64,
            plan_factor_safety: env_f("PLAN_FACTOR_SAFETY", 0.97),
            required_bonus: env_f("REQUIRED_BONUS", 80.0),
            req_p_lo: env_f("REQ_P_LO", 0.95),
            req_p_hi: env_f("REQ_P_HI", 1.35),
            request_mult: env_f("REQUEST_MULT", 3.0),
            request_floor: env_f("REQUEST_FLOOR", 0.0),
            req_calib_power: env_f("REQ_CALIB_POWER", 0.0),
            forecast_discount: env_f("FORECAST_DISCOUNT", 0.2),
            req_calendar: env_i("REQ_CALENDAR", 0) != 0,
            req_timing: env_f("REQ_TIMING", 0.85),
            req_timing_nights: env_i("REQ_TIMING_NIGHTS", 5),
            req_timing_discount: env_f("REQ_TIMING_DISCOUNT", 0.3),
            urgency: env_f("URGENCY", 1.5),
            partial_discount: env_f("PARTIAL_DISCOUNT", 1.0),
            partial_done: env_f("PARTIAL_DONE", 0.9),
            partial_nights: env_i("PARTIAL_NIGHTS", 3),
            plan_util: env_f("PLAN_UTIL", 0.55),
            plan_q: env_f("PLAN_Q", 0.7),
            kappa: env_f("KAPPA", 1.0),
            band_opt: env_f("BAND_OPT", 1.0),
            band_half_life: env_f("BAND_HALF_LIFE", 0.0),
            band_fallback: env_i("BAND_FALLBACK", 0) != 0,
            band_fallback_hours: env_f("BAND_FALLBACK_HOURS", 12.0),
            band_cont: env_i("BAND_CONT", 0) != 0,
            weather_gate: env_i("WEATHER_GATE", 0) != 0,
        }
    }
}

/// A JSON number (or numeric string) as f64.
pub fn num(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap_or(0.0),
        Value::String(s) => s.trim().parse().unwrap_or(0.0),
        Value::Bool(b) => f64::from(u8::from(*b)),
        _ => 0.0,
    }
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().unwrap_or(0.0) != 0.0,
        Value::String(s) => !s.is_empty(),
        Value::Null => false,
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn str_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values[values.len() / 2]
}

pub const PROGRAMS: [&str; 3] = ["DARK", "BRIGHT", "BACKUP"];

/// Prediction for one assigned target of the observe in flight.
#[derive(Clone)]
struct Pred {
    model: f64,
    band_model: f64,
    alt: f64,
    az: f64,
    pred: f64,
    dir_clean: bool,
    fiber: Option<usize>,
}

/// (hours, model, program, matched, direction-clean) of a saturated hit.
type BandObs = (f64, f64, usize, bool, bool);
/// (alt, az, fiber, hit) of one assigned target.
type OffsetRow = (f64, f64, usize, bool);

/// Per-target geometry at the start of the exposure: (alt, az, lunar, seconds up, gain multiplier).
#[derive(Clone, Copy)]
struct Base {
    alt: f64,
    az: f64,
    lunar: f64,
    up: f64,
    mult: f64,
}

/// Base plus the sky model at the start and 30 minutes later.
#[derive(Clone, Copy)]
struct Info {
    m0: f64,
    m1: f64,
    up: f64,
    mult: f64,
    alt: f64,
    az: f64,
}

/// Best (net, T, pick, total) for one pointing.
struct Found {
    net: f64,
    t: f64,
    pick: Vec<(usize, usize)>,
    total: f64,
}

pub struct Planner {
    p: Params,
    lat: f64,
    lon: f64,
    pub min_alt: f64,
    pub nights: Vec<(f64, f64)>,
    survey_end: f64,
    pub slot_seconds: i64,
    grid: FiberGrid,
    pub min_exposure: f64,
    max_exposure: f64,
    f0t0: f64,
    q0: f64,
    airmass_exponent: f64,
    band_dark: f64,
    band_bright: f64,
    multipliers: [f64; 3],
    mismatch: f64,
    lunar_model: LunarModel,
    pub ids: Vec<String>,
    index_of: HashMap<String, usize>,
    ra: Vec<f64>,
    dec: Vec<f64>,
    flux: Vec<f64>,
    weight: Vec<f64>,
    pub required: Vec<bool>,
    cos_dec: Vec<f64>,
    hmax: Vec<f64>,
    factor: Vec<f64>,
    cur: Vec<f64>,
    rate_ema: f64,
    /// (hour, night, [E samples]): quality level / band level.
    pub e_hours: Vec<(i64, usize, Vec<f64>)>,
    e_ratios: VecDeque<(f64, f64)>,
    misses: Vec<i64>,
    vcache: Option<Vec<f64>>,
    vdirty: HashSet<usize>,
    attempts: Vec<i64>,
    active: Vec<usize>,
    alt_a: Vec<f64>,
    alt_b: Vec<f64>,
    alt_c: Vec<f64>,
    sin_alt_limit: f64,
    cells: HashMap<i64, Vec<(f64, usize)>>,
    cell_ras: HashMap<i64, Vec<f64>>,
    first_night: Vec<i64>,
    last_night: Vec<i64>,
    ideal_model: Vec<f64>,
    night_best: HashMap<usize, Vec<f64>>,
    scarcity: f64,
    lambda_frac: f64,
    pub scale: f64,
    prior_scale: f64,
    samples: VecDeque<(f64, f64)>,
    all_ratios: VecDeque<f64>,
    pending_cmd: Option<(f64, f64)>,
    pub bad_forecast: bool,
    req_calib: HashMap<usize, f64>,
    offset: (f64, f64),
    offset_evidence: VecDeque<((f64, f64), Vec<OffsetRow>)>,
    offset_scores: Option<Vec<i64>>,
    offset_step: f64,
    offset_steps: i64,
    offset_grid: Vec<(f64, f64)>,
    offset_misses: i64,
    offset_updates: i64,
    pub band_level: Option<f64>,
    band_obs: VecDeque<BandObs>,
    pending: Vec<(usize, Pred)>,
    pending_program: usize,
    pending_duration: f64,
    blocked: Vec<(f64, f64)>,
    /// (event kind, direction) of the current bulletin, terrain excluded.
    pub notices: BTreeSet<(String, String)>,
    terrain: BTreeSet<String>,
    pub extra_avoid: BTreeSet<String>,
    pub fast_level: usize,
    request_bonus: HashMap<usize, f64>,
    request_threshold: HashMap<usize, f64>,
    density_order: Vec<usize>,
    planned: Vec<bool>,
    plan_night: i64,
    dbg_none_hour: Option<i64>,
}

fn push_bounded<T>(deque: &mut VecDeque<T>, item: T, maxlen: usize) {
    if deque.len() == maxlen {
        deque.pop_front();
    }
    deque.push_back(item);
}

impl Planner {
    pub fn new(init: &Value) -> Planner {
        let p = Params::load();
        let site = &init["site"];
        let lat = num(&site["latitude_deg"]);
        let lon = num(&site["longitude_deg"]);
        let min_alt = num(&site["minimum_altitude_deg"]);
        let survey = &init["survey"];
        let nights: Vec<(f64, f64)> = survey["nights"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|n| (parse_utc(n["observing_start_utc"].as_str().unwrap_or("")), parse_utc(n["observing_end_utc"].as_str().unwrap_or(""))))
                    .collect()
            })
            .unwrap_or_default();
        let survey_end = parse_utc(survey["end_utc"].as_str().unwrap_or(""));
        let slot_seconds = num(&survey["slot_seconds"]) as i64;
        let instrument = &init["instrument"];
        let grid = FiberGrid::new(instrument);
        let min_exposure = num(&instrument["exposure"]["min_duration_seconds"]).trunc();
        let max_exposure = num(&instrument["exposure"]["max_duration_seconds"]).trunc();
        let score = &init["scoring"];
        let f0t0 = num(&score["flux_zero_point"]) * num(&score["exposure_zero_point_seconds"]);
        let q0 = num(&score["q0"]);
        let airmass_exponent = num(&score["airmass_exponent"]);
        let program = &score["program"];
        let multipliers = [
            num(&program["multipliers"]["DARK"]),
            num(&program["multipliers"]["BRIGHT"]),
            num(&program["multipliers"]["BACKUP"]),
        ];
        let lm = &score["lunar_model"];
        let lunar_model = LunarModel {
            maximum_penalty: num(&lm["maximum_penalty"]),
            altitude_exponent: num(&lm["altitude_exponent"]),
            angular_decay_scale_deg: num(&lm["angular_decay_scale_deg"]),
        };

        let columns: Vec<String> = init["targets"]["columns"].as_array().map(|a| a.iter().map(str_of).collect()).unwrap_or_default();
        let col = |name: &str| columns.iter().position(|c| c == name).unwrap_or(usize::MAX);
        let (c_id, c_ra, c_dec, c_flux, c_w, c_req) =
            (col("target_id"), col("ra_deg"), col("dec_deg"), col("feature_flux"), col("science_weight"), col("required"));
        let empty = Vec::new();
        let rows = init["targets"]["rows"].as_array().unwrap_or(&empty);
        let cell = |row: &Value, c: usize| row.get(c).cloned().unwrap_or(Value::Null);
        let ids: Vec<String> = rows.iter().map(|r| str_of(&cell(r, c_id))).collect();
        let index_of = ids.iter().enumerate().map(|(i, id)| (id.clone(), i)).collect();
        let ra: Vec<f64> = rows.iter().map(|r| num(&cell(r, c_ra))).collect();
        let dec: Vec<f64> = rows.iter().map(|r| num(&cell(r, c_dec))).collect();
        let flux: Vec<f64> = rows.iter().map(|r| num(&cell(r, c_flux))).collect();
        let weight: Vec<f64> = rows.iter().map(|r| num(&cell(r, c_w))).collect();
        let required: Vec<bool> = rows.iter().map(|r| truthy(&cell(r, c_req))).collect();
        let n = rows.len();
        let sin_dec: Vec<f64> = dec.iter().map(|d| d.to_radians().sin()).collect();
        let cos_dec: Vec<f64> = dec.iter().map(|d| d.to_radians().cos()).collect();
        let sin_lat = lat.to_radians().sin();
        let cos_lat = lat.to_radians().cos();
        let hmax: Vec<f64> = dec.iter().map(|&d| max_hour_angle_deg(d, lat, min_alt + ALT_MARGIN_DEG)).collect();
        let active: Vec<usize> = (0..n).filter(|&i| hmax[i] > 0.0).collect();
        let alt_a = sin_dec.iter().map(|s| sin_lat * s).collect();
        let alt_b = (0..n).map(|i| cos_lat * cos_dec[i] * ra[i].to_radians().cos()).collect();
        let alt_c = (0..n).map(|i| cos_lat * cos_dec[i] * ra[i].to_radians().sin()).collect();
        // best sky model a target can ever get: at transit, Moon down (used to time faint required targets)
        let ideal_model = dec
            .iter()
            .map(|&d| 1.0 / (q0 * normalized_airmass((90.0 - (d - lat).abs()).max(1.0)).powf(airmass_exponent)))
            .collect();
        // Time scarcity: fibre-seconds the catalogue needs (typical sky) / night seconds on offer. A time-rich
        // season should price telescope time lower than a tight one.
        let night_seconds = psum(nights.iter().map(|(s, e)| e - s));
        let need = psum(flux.iter().map(|&f| max_exposure.min(f0t0 / (f.max(1e-3) * TYPICAL_Q)))) / grid.n as f64;
        let scarcity = need / night_seconds.max(1.0);
        let lambda_frac = p.lambda_frac * 1.0f64.min((scarcity / p.scarcity_ref).max(0.2).powf(p.scarcity_power));
        log(&format!("planner: scarcity {:.2}, time price fraction {:.2}", scarcity, lambda_frac));
        let mut density_order: Vec<usize> = (0..n).collect();
        // Python's stable sort by -w*flux
        density_order.sort_by(|&a, &b| (-weight[a] * flux[a]).partial_cmp(&(-weight[b] * flux[b])).unwrap_or(std::cmp::Ordering::Equal));
        let offset_step = grid.pitch / 12.0;
        let mut planner = Planner {
            lat,
            lon,
            min_alt,
            survey_end,
            slot_seconds,
            min_exposure,
            max_exposure,
            f0t0,
            q0,
            airmass_exponent,
            band_dark: num(&program["bands"]["DARK"]),
            band_bright: num(&program["bands"]["BRIGHT"]),
            multipliers,
            mismatch: num(&program["mismatch_multiplier"]),
            lunar_model,
            ids,
            index_of,
            factor: vec![0.0; n],
            cur: vec![0.0; n],
            rate_ema: 0.0,
            e_hours: Vec::new(),
            e_ratios: VecDeque::new(),
            misses: vec![0; n],
            vcache: None,
            vdirty: HashSet::new(),
            attempts: vec![0; n],
            active,
            alt_a,
            alt_b,
            alt_c,
            sin_alt_limit: (min_alt + ALT_MARGIN_DEG).to_radians().sin(),
            cells: HashMap::new(),
            cell_ras: HashMap::new(),
            first_night: Vec::new(),
            last_night: Vec::new(),
            ideal_model,
            night_best: HashMap::new(),
            scarcity,
            lambda_frac,
            scale: 1.0,
            prior_scale: 1.0,
            samples: VecDeque::new(),
            all_ratios: VecDeque::new(),
            pending_cmd: None,
            bad_forecast: false,
            req_calib: HashMap::new(),
            offset: (0.0, 0.0),
            offset_evidence: VecDeque::new(),
            offset_scores: None,
            offset_step,
            offset_steps: OFFSET_STEPS,
            offset_grid: Vec::new(),
            offset_misses: 0,
            offset_updates: 0,
            band_level: None,
            band_obs: VecDeque::new(),
            pending: Vec::new(),
            pending_program: 2,
            pending_duration: 0.0,
            blocked: Vec::new(),
            notices: BTreeSet::new(),
            terrain: BTreeSet::new(),
            extra_avoid: BTreeSet::new(),
            fast_level: 0,
            request_bonus: HashMap::new(),
            request_threshold: HashMap::new(),
            density_order,
            planned: vec![false; n],
            plan_night: -1,
            dbg_none_hour: None,
            ra,
            dec,
            flux,
            weight,
            required,
            cos_dec,
            hmax,
            nights,
            grid,
            p,
        };
        planner.build_index();
        planner.build_windows();
        planner.build_required_calendar();
        planner.offset_grid = planner.make_offset_grid();
        let _ = planner.scarcity;
        planner
    }

    // --- precomputation --------------------------------------------------------------------------

    fn build_index(&mut self) {
        self.cells.clear();
        for &i in &self.active {
            self.cells.entry(self.dec[i].floor() as i64).or_default().push((self.ra[i], i));
        }
        for band in self.cells.values_mut() {
            band.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }
        self.cell_ras = self.cells.iter().map(|(k, band)| (*k, band.iter().map(|(ra, _)| *ra).collect())).collect();
    }

    /// Per night, the best public sky model (airmass + Moon, no weather) a required target can get.
    /// Uses only geometry and the lunar ephemeris: when the target is highest during that night's window.
    fn build_required_calendar(&mut self) {
        self.night_best.clear();
        if !self.p.req_calendar {
            return;
        }
        let lsts: Vec<(f64, f64, f64)> = self.nights.iter().map(|&(s, e)| (s, local_sidereal_deg(s, self.lon), e - s)).collect();
        let mut moons: HashMap<(usize, i64), Moon> = HashMap::new();
        for i in 0..self.ids.len() {
            if !self.required[i] || self.hmax[i] <= 0.0 {
                continue;
            }
            let mut row = Vec::new();
            for (k, &(start, l0, span)) in lsts.iter().enumerate() {
                // seconds after night start when hour angle is closest to 0
                let ha0 = wrap180(l0 - self.ra[i]);
                let t = (-ha0 / SIDEREAL_DEG_PER_SECOND).max(0.0).min(span);
                let ha = wrap180(ha0 + t * SIDEREAL_DEG_PER_SECOND);
                if ha.abs() > self.hmax[i] {
                    row.push(0.0);
                    continue;
                }
                let key = (k, (t / 1800.0).floor() as i64);
                let moon = moons.entry(key).or_insert_with(|| {
                    let moment = start + (key.1 as f64 + 0.5) * 1800.0;
                    Moon::new(moment, local_sidereal_deg(moment, self.lon), self.lat, self.lunar_model)
                });
                let (alt, _) = radec_to_altaz(self.ra[i], self.dec[i], l0 + t * SIDEREAL_DEG_PER_SECOND, self.lat);
                row.push(moon.lunar_factor(self.ra[i], self.dec[i]) / (self.q0 * normalized_airmass(alt.max(1.0)).powf(self.airmass_exponent)));
            }
            self.night_best.insert(i, row);
        }
    }

    fn best_future_model(&self, i: usize, night_index: usize) -> f64 {
        match self.night_best.get(&i) {
            Some(row) if night_index + 1 < row.len() => row[night_index + 1..].iter().cloned().fold(f64::MIN, f64::max),
            _ => 0.0,
        }
    }

    fn neighbours(&self, ra: f64, dec: f64, radius: f64) -> Vec<usize> {
        let mut found = Vec::new();
        let cos_dec = (89.0f64.min(dec.abs() + radius)).to_radians().cos().max(0.05);
        let width = radius / cos_dec;
        for key in ((dec - radius).floor() as i64)..=((dec + radius).floor() as i64) {
            let Some(band) = self.cells.get(&key) else { continue };
            if band.is_empty() {
                continue;
            }
            let ras = &self.cell_ras[&key];
            let mut spans = vec![(ra - width, ra + width)];
            if spans[0].0 < 0.0 {
                spans = vec![(0.0, spans[0].1), (spans[0].0 + 360.0, 360.0)];
            } else if spans[0].1 >= 360.0 {
                spans = vec![(spans[0].0, 360.0), (0.0, spans[0].1 - 360.0)];
            }
            for (low, high) in spans {
                let a = ras.partition_point(|&x| x < low);
                let b = ras.partition_point(|&x| x <= high);
                for k in a..b.max(a) {
                    found.push(band[k].1);
                }
            }
        }
        found
    }

    /// First and last night on which each target has at least 20 minutes above the limit.
    fn build_windows(&mut self) {
        let need = 20.0 * 60.0 * SIDEREAL_DEG_PER_SECOND;
        let spans: Vec<(f64, f64)> =
            self.nights.iter().map(|&(s, e)| (local_sidereal_deg(s, self.lon), (e - s) * SIDEREAL_DEG_PER_SECOND)).collect();
        self.first_night = vec![self.nights.len() as i64; self.ra.len()];
        self.last_night = vec![-1; self.ra.len()];
        for &i in &self.active {
            let h = self.hmax[i];
            for (k, &(l0, span)) in spans.iter().enumerate() {
                let overlap = if h >= 180.0 {
                    span
                } else {
                    let a = pmod(self.ra[i] - h - l0, 360.0);
                    (span.min(a + 2.0 * h) - a).max(0.0) + span.min(a - 360.0 + 2.0 * h).max(0.0)
                };
                if overlap >= need {
                    if self.first_night[i] > k as i64 {
                        self.first_night[i] = k as i64;
                    }
                    self.last_night[i] = k as i64;
                }
            }
        }
    }

    // --- messages and results --------------------------------------------------------------------

    pub fn on_messages(&mut self, messages: &[Value], latest_bulletin: &Value) {
        for message in messages {
            let kind = message.get("record_type").and_then(Value::as_str).unwrap_or("");
            if kind == "bulletin" && message.get("initial").map_or(false, truthy) {
                for notice in message.get("notices").and_then(Value::as_array).into_iter().flatten() {
                    if notice.get("event_kind").and_then(Value::as_str) == Some("terrain_obstruction") {
                        self.terrain.insert(notice.get("direction").map(str_of).unwrap_or_default());
                    }
                }
            } else if kind == "state_resync" {
                self.resync(message);
            }
        }
        self.notices = latest_bulletin
            .get("notices")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|n| n.get("event_kind").and_then(Value::as_str) != Some("terrain_obstruction"))
            .map(|n| (n.get("event_kind").map(str_of).unwrap_or_default(), n.get("direction").map(str_of).unwrap_or_default()))
            .collect();
    }

    /// Turn the current all-or-nothing request rewards into per-target planning values.
    pub fn on_requests(&mut self, requests: &[Value]) {
        let old = std::mem::take(&mut self.request_bonus);
        self.request_threshold = HashMap::new();
        for request in requests {
            // A request that already met its minimum still appears until its deadline
            // with remaining_count 0; its reward is settled, so it adds no value.
            let remaining = num(request.get("remaining_count").unwrap_or(&request["minimum_completed"])).trunc() as i64;
            if remaining <= 0 {
                continue;
            }
            let completed: HashSet<String> =
                request.get("completed_target_ids").and_then(Value::as_array).into_iter().flatten().map(str_of).collect();
            // the settlement (+500 on every observed card) is granted separately from completion_reward,
            // which the new cards set to 0: value each request at the floor so the engine still chases it
            let reward = num(&request["completion_reward"]).max(self.p.request_floor);
            let unit = self.p.request_mult * reward / remaining as f64;
            let threshold = num(&request["completion_factor_threshold"]);
            for target in request.get("target_ids").and_then(Value::as_array).into_iter().flatten() {
                let target_id = str_of(target);
                if completed.contains(&target_id) {
                    continue;
                }
                let Some(&i) = self.index_of.get(&target_id) else { continue };
                // Overlapping requests: the marginal rewards add up; the combined gain is
                // only collectible at the highest threshold of the contributing requests.
                *self.request_bonus.entry(i).or_insert(0.0) += unit;
                let t = self.request_threshold.entry(i).or_insert(0.0);
                *t = t.max(threshold);
                if self.hmax[i] > 0.0 && !self.active.contains(&i) {
                    self.active.push(i);
                }
            }
        }
        if old != self.request_bonus {
            self.vdirty.extend(old.keys());
            self.vdirty.extend(self.request_bonus.keys());
        }
    }

    /// Part of the recent data was lost: restart the factor estimates from the engine's best scores.
    fn resync(&mut self, message: &Value) {
        let best: HashMap<String, f64> = message
            .get("best_scores")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|row| (str_of(&row["target_id"]), num(&row["best_score"])))
            .collect();
        let top = self.multipliers.iter().cloned().fold(f64::MIN, f64::max);
        for i in 0..self.ids.len() {
            let score = best.get(&self.ids[i]).cloned().unwrap_or(0.0);
            self.factor[i] = if score > 0.0 { (score / (self.weight[i] * top)).min(1.0) } else { 0.0 };
            self.cur[i] = score / self.weight[i];
        }
        self.active = (0..self.ids.len()).filter(|&i| self.hmax[i] > 0.0).collect();
        self.vcache = None;
        self.pending.clear();
        log(&format!("state_resync: {} targets keep a score; plan rebuilt", best.len()));
    }

    pub fn site_closed(&self) -> bool {
        self.notices.iter().any(|(k, d)| CLOSED_KINDS.contains(&k.as_str()) && d == "ALL")
    }

    fn all_sky_weather(&self) -> bool {
        self.notices.iter().any(|(k, d)| d == "ALL" && SKY_WEATHER_KINDS.contains(&k.as_str()))
    }

    /// Update factor estimates and the sky-quality estimate from the previous observe.
    pub fn on_result(&mut self, result: &Value, hours: f64) {
        if result.get("action").and_then(Value::as_str) != Some("observe") || self.pending.is_empty() {
            self.pending.clear();
            return;
        }
        let mut hits: HashMap<String, f64> = HashMap::new();
        for hit in result.get("hits").and_then(Value::as_array).into_iter().flatten() {
            hits.insert(str_of(&hit["target_id"]), num(&hit["score"]));
        }
        let any_positive = hits.values().any(|&s| s > 0.0);
        let declared = self.multipliers[self.pending_program];
        let pending = std::mem::take(&mut self.pending);
        for (i, prediction) in &pending {
            let i = *i;
            self.vdirty.insert(i);
            let Some(&score) = hits.get(&self.ids[i]) else {
                self.misses[i] += 1; // a miss: the target did not land on its fibre glass
                continue;
            };
            if score <= 0.0 {
                if any_positive {
                    self.blocked.push((prediction.az, prediction.alt));
                }
                continue;
            }
            // score = weight * factor * multiplier; the multiplier is `declared` if the program matched.
            // A saturated hit (factor = 1) shows the multiplier exactly, so it tells whether the sky's
            // program band matched the declared program. Instrument efficiency does not enter the band.
            let multiplier_seen = score / self.weight[i];
            self.cur[i] = self.cur[i].max(multiplier_seen);
            if (multiplier_seen - declared).abs() < 2e-4 {
                push_bounded(&mut self.band_obs, (hours, prediction.model, self.pending_program, true, prediction.dir_clean), 300);
            } else if (multiplier_seen - self.mismatch).abs() < 2e-4 {
                push_bounded(&mut self.band_obs, (hours, prediction.model, self.pending_program, false, prediction.dir_clean), 300);
            }
            let factor_if_match = score / (self.weight[i] * declared);
            let factor_if_miss = score / (self.weight[i] * self.mismatch);
            let ratio_match = factor_if_match * self.f0t0 / (self.flux[i] * self.pending_duration * prediction.model);
            let matched = self.band(ratio_match * prediction.band_model) == self.pending_program;
            let factor = if matched { factor_if_match } else { factor_if_miss };
            self.factor[i] = self.factor[i].max(factor.min(1.0));
            if self.required[i] && self.factor[i] < 0.5 {
                let pred = prediction.pred;
                if pred > 0.0 && factor < 0.97 {
                    self.req_calib.insert(i, (factor / pred).max(0.3).min(1.0).powf(self.p.req_calib_power));
                }
                self.attempts[i] += 1; // not enough yet: lower its priority a little for next time
            }
            if factor < 0.97 {
                let ratio = factor * self.f0t0 / (self.flux[i] * self.pending_duration * prediction.model);
                push_bounded(&mut self.samples, (hours, ratio), 24);
                push_bounded(&mut self.all_ratios, ratio, 400);
                if prediction.dir_clean {
                    push_bounded(&mut self.e_ratios, (hours, ratio), 400);
                }
            }
        }
        self.offset_evidence_update(&pending, &hits);
        self.update_scale(hours);
    }

    // --- pointing offset (Hard-mode cards) ----------------------------------------------------------------

    /// Hard-mode cards add a hidden fixed offset to every pointing (participant guide: actual = command +
    /// (d_alt, d_az)). Each assigned target's hit or miss is evidence; keep a score for each candidate offset
    /// on a grid and adopt the best one once it clearly explains the misses better than no offset.
    fn offset_evidence_update(&mut self, pending: &[(usize, Pred)], hits: &HashMap<String, f64>) {
        let Some(cmd) = self.pending_cmd else { return };
        if pending.is_empty() {
            return;
        }
        let rows: Vec<OffsetRow> = pending
            .iter()
            .filter_map(|(i, p)| p.fiber.map(|f| (p.alt, p.az, f, hits.contains_key(&self.ids[*i]))))
            .collect();
        if rows.is_empty() {
            return;
        }
        let missed = rows.iter().filter(|r| !r.3).count() as i64;
        self.offset_misses += missed;
        push_bounded(&mut self.offset_evidence, (cmd, rows.clone()), 400);
        if self.offset_misses < OFFSET_MIN_MISSES {
            return;
        }
        if self.offset_scores.is_none() {
            let evidence: Vec<_> = self.offset_evidence.iter().cloned().collect();
            self.rescore_offsets(&evidence);
        } else {
            self.score_offsets(cmd, &rows);
        }
        self.offset_updates += 1;
        if self.offset_updates % OFFSET_REFINE_EVERY == 1 || OFFSET_REFINE_EVERY <= 1 {
            self.refine_offset();
        }
    }

    fn consistent(&self, cmd: (f64, f64), rows: &[OffsetRow], d_alt: f64, d_az: f64) -> i64 {
        let (c_alt, c_az) = (cmd.0 + d_alt, pmod(cmd.1 + d_az, 360.0));
        let mut ok = 0;
        for &(t_alt, t_az, fiber, hit) in rows {
            let fib = tangent_offsets(t_alt, t_az, c_alt, c_az).and_then(|(n, e)| self.grid.classify(n, e).0);
            if (fib == Some(fiber)) == hit {
                ok += 1;
            }
        }
        ok
    }

    fn make_offset_grid(&self) -> Vec<(f64, f64)> {
        let (n, step) = (self.offset_steps, self.offset_step);
        let mut grid = Vec::new();
        for i in -n..=n {
            for j in -n..=n {
                grid.push((step * i as f64, step * j as f64));
            }
        }
        grid
    }

    fn rescore_offsets(&mut self, evidence: &[((f64, f64), Vec<OffsetRow>)]) {
        self.offset_scores = Some(vec![0; self.offset_grid.len()]);
        for (cmd, past) in evidence {
            self.score_offsets(*cmd, past);
        }
    }

    fn score_offsets(&mut self, cmd: (f64, f64), rows: &[OffsetRow]) {
        let add: Vec<i64> = self.offset_grid.iter().map(|&(a, z)| self.consistent(cmd, rows, a, z)).collect();
        if let Some(scores) = self.offset_scores.as_mut() {
            for (s, a) in scores.iter_mut().zip(add) {
                *s += a;
            }
        }
    }

    fn best_offset_index(&self) -> usize {
        let scores = self.offset_scores.as_ref().expect("offset scores");
        let mut k = 0;
        for (n, &s) in scores.iter().enumerate() {
            if s > scores[k] {
                k = n;
            }
        }
        k
    }

    fn refine_offset(&mut self) {
        let k = self.best_offset_index();
        let (mut base_alt, mut base_az) = self.offset_grid[k];
        let n = self.offset_steps;
        let edge = base_alt.abs().max(base_az.abs()) >= (n as f64 - 0.5) * self.offset_step;
        if edge && (n + 1) as f64 * self.offset_step < self.grid.fov / 2.0 {
            // the best candidate sits on the edge of the grid: the offset may be larger, widen the search
            self.offset_steps = (n as f64 * 1.5) as i64 + 1;
            self.offset_grid = self.make_offset_grid();
            let len = self.offset_evidence.len();
            let recent: Vec<_> = self.offset_evidence.iter().skip(len.saturating_sub(100)).cloned().collect();
            self.rescore_offsets(&recent);
            log(&format!("pointing offset: search widened to +-{:.2} deg", self.offset_steps as f64 * self.offset_step));
            let k = self.best_offset_index();
            (base_alt, base_az) = self.offset_grid[k];
        }
        let len = self.offset_evidence.len();
        let evidence: Vec<_> = self.offset_evidence.iter().skip(len.saturating_sub(OFFSET_FINE_EVIDENCE)).collect();
        let zero: i64 = evidence.iter().map(|(cmd, rows)| self.consistent(*cmd, rows, 0.0, 0.0)).sum();
        let mut scored = Vec::new();
        let fine = self.offset_step / 5.0;
        for i in -6..=6 {
            for j in -6..=6 {
                let (d_alt, d_az) = (base_alt + fine * i as f64, base_az + fine * j as f64);
                let s: i64 = evidence.iter().map(|(cmd, rows)| self.consistent(*cmd, rows, d_alt, d_az)).sum();
                scored.push((s, d_alt, d_az));
            }
        }
        let top = scored.iter().map(|s| s.0).max().unwrap_or(0);
        if top - zero < OFFSET_MARGIN {
            return;
        }
        // several offsets often explain the evidence equally well: take the centre of that set
        let tied: Vec<(f64, f64)> = scored.iter().filter(|s| s.0 == top).map(|s| (s.1, s.2)).collect();
        let count = tied.len() as f64;
        let centre = (
            round_to(psum(tied.iter().map(|t| t.0)) / count, 3),
            round_to(psum(tied.iter().map(|t| t.1)) / count, 3),
        );
        if (centre.0 - self.offset.0).abs() + (centre.1 - self.offset.1).abs() < 0.005 {
            return;
        }
        let total: usize = evidence.iter().map(|(_, rows)| rows.len()).sum();
        log(&format!(
            "pointing offset: alt {:+.3} az {:+.3} deg explains {}/{} fibre outcomes (no offset: {}; {} equally good grid points)",
            centre.0, centre.1, top, total, zero, tied.len()
        ));
        if self.offset == (0.0, 0.0) {
            self.misses = vec![0; self.ids.len()]; // the misses were the offset, not the targets
            self.vcache = None;
        }
        self.offset = centre;
    }

    /// Sky level for the program band, fitted to recent saturated hits (they show the multiplier exactly).
    /// Falls back to the quality-based estimate; an unreported instrument fault lowers quality but not the band.
    fn calibrated_band_scale(&mut self, hours: f64) -> f64 {
        let guess = self.scale / 0.95;
        let mut recent: Vec<BandObs> = self.band_obs.iter().filter(|o| o.0 >= hours - BAND_MEMORY_HOURS).cloned().collect();
        if recent.len() < 4 && self.p.band_fallback {
            // few saturated hits lately (poor quality, or an instrument fault): the band does not follow the
            // instrument, so keep fitting the latest saturated hits instead of the quality level
            let all: Vec<BandObs> = self.band_obs.iter().filter(|o| o.0 >= hours - self.p.band_fallback_hours).cloned().collect();
            recent = all[all.len().saturating_sub(8)..].to_vec();
        }
        if recent.len() < 4 {
            return guess;
        }
        // among equally consistent levels prefer the one closest to the last fitted level (the sky band moves
        // with the weather, not with the instrument), else to the quality-based guess
        let centre = match self.band_level {
            Some(level) if self.p.band_cont && level != 0.0 => level,
            _ => guess,
        };
        let weights: Vec<f64> = if self.p.band_half_life > 0.0 {
            recent.iter().map(|o| 0.5f64.powf((hours - o.0) / self.p.band_half_life)).collect()
        } else {
            vec![1.0; recent.len()]
        };
        let mut best: Option<((f64, i64), f64)> = None;
        for step in -25i64..31 {
            let w = centre * 1.06f64.powf(step as f64);
            let ok = psum(weights.iter().zip(&recent).filter(|(_, o)| (self.band(o.1 * w) == o.2) == o.3).map(|(wt, _)| *wt));
            let key = (round_to(ok, 6), -step.abs());
            let better = match &best {
                None => true,
                Some((bk, _)) => key.0 > bk.0 || (key.0 == bk.0 && key.1 > bk.1),
            };
            if better {
                best = Some((key, w));
            }
        }
        let level = best.map(|b| b.1).unwrap_or(guess);
        self.band_level = Some(level);
        level
    }

    /// E = clean-sky quality level / clean-sky band level over the last BAND_MEMORY_HOURS (1 = consistent).
    fn clean_e(&self, hours: f64) -> Option<f64> {
        let mut ratios: Vec<f64> = self.e_ratios.iter().filter(|(h, _)| *h >= hours - BAND_MEMORY_HOURS).map(|(_, r)| *r).collect();
        let obs: Vec<&BandObs> = self.band_obs.iter().filter(|o| o.0 >= hours - BAND_MEMORY_HOURS && o.4).collect();
        if ratios.len() < 4 || obs.len() < 4 {
            return None;
        }
        let guess = median(&mut ratios) / 0.95;
        let mut best: Option<((i64, i64), f64)> = None;
        for step in -20i64..31 {
            let w = guess * 1.06f64.powf(step as f64);
            let ok = obs.iter().filter(|o| (self.band(o.1 * w) == o.2) == o.3).count() as i64;
            let key = (ok, -step.abs());
            if best.as_ref().map_or(true, |b| key > b.0) {
                best = Some((key, w));
            }
        }
        Some((guess / best.unwrap().1).min(1.0))
    }

    /// Sky quality now = median of recent samples; fall back to the long-run median when stale.
    pub fn update_scale(&mut self, hours: f64) {
        if self.all_ratios.len() >= 8 {
            let mut ordered: Vec<f64> = self.all_ratios.iter().cloned().collect();
            self.prior_scale = median(&mut ordered);
        }
        let mut recent: Vec<f64> = self.samples.iter().filter(|(w, _)| *w >= hours - SKY_MEMORY_HOURS).map(|(_, r)| *r).collect();
        self.scale = if recent.len() >= 4 { median(&mut recent).max(0.05) } else { self.prior_scale };
    }

    fn band(&self, q_band: f64) -> usize {
        if q_band >= self.band_dark {
            0
        } else if q_band >= self.band_bright {
            1
        } else {
            2
        }
    }

    // --- anomaly check ---------------------------------------------------------------------------

    /// After a correct report the instrument is repaired: start the quality estimates afresh.
    pub fn forget_quality_history(&mut self) {
        self.e_hours.clear();
        self.e_ratios.clear();
        self.samples.clear();
        self.all_ratios.clear();
        self.prior_scale = 1.0;
    }

    // --- planning --------------------------------------------------------------------------------

    pub fn current_night(&self, now: f64) -> Option<(usize, f64, f64)> {
        self.nights.iter().enumerate().find(|(_, &(s, e))| s <= now && now < e).map(|(k, &(s, e))| (k, s, e))
    }

    pub fn next_night_start(&self, now: f64) -> Option<f64> {
        self.nights.iter().map(|&(s, _)| s).find(|&s| s > now)
    }

    fn direction_factor(&self, alt: f64, az: f64) -> f64 {
        let mut factor: f64 = 1.0;
        for direction in &self.terrain {
            if let Some(daz) = direction_az(direction) {
                if alt < 50.0 && az_distance(az, daz) <= 60.0 {
                    return 0.0;
                }
            }
        }
        for (kind, direction) in &self.notices {
            let Some(daz) = direction_az(direction) else { continue };
            let near = az_distance(az, daz) <= 67.5;
            if BLOCKING_KINDS.contains(&kind.as_str()) && near && alt < 62.0 {
                return 0.0;
            }
            if near && alt < 75.0 {
                factor = factor.min(0.35);
            }
        }
        for direction in &self.extra_avoid {
            if let Some(daz) = direction_az(direction) {
                if az_distance(az, daz) <= 67.5 && alt < 70.0 {
                    factor = factor.min(0.35);
                }
            }
        }
        let start = self.blocked.len().saturating_sub(40);
        for &(blocked_az, blocked_alt) in &self.blocked[start..] {
            if az_distance(az, blocked_az) <= 12.0 && alt <= blocked_alt + 3.0 {
                factor = factor.min(0.2);
            }
        }
        factor
    }

    fn value(&self, i: usize) -> f64 {
        let f = self.factor[i];
        let damp = 0.6f64.powf(self.misses[i] as f64);
        let request = self.request_bonus.get(&i).cloned().unwrap_or(0.0);
        if self.required[i] {
            if f >= REQUIRED_SAFE_FACTOR {
                return (self.weight[i] * (1.0 - f * f).max(0.0) + request) * damp;
            }
            return (self.weight[i] * (1.0 - f * f) + self.p.required_bonus * (if f < 0.5 { 1.0 } else { 0.35 }) + request) * damp;
        }
        ((if f >= DONE_FACTOR { 0.0 } else { self.weight[i] * (1.0 - f * f) }) + request) * damp
    }

    /// Which targets will the season complete? Fill the remaining useful fibre-time with targets in
    /// order of value density (w * flux); the rest are fillers whose partial exposures are worth taking.
    fn season_plan(&mut self, now: f64, night_index: usize) {
        self.plan_night = night_index as i64;
        let mut cap = psum(self.nights.iter().filter(|&&(_, e)| e > now).map(|&(s, e)| (e - s.max(now)).max(0.0)));
        cap *= self.grid.n as f64 * self.p.plan_util;
        self.planned = vec![false; self.ids.len()];
        let mut n = 0;
        for &i in &self.density_order {
            if cap <= 0.0 {
                break;
            }
            if self.hmax[i] <= 0.0 || self.factor[i] >= DONE_FACTOR {
                continue;
            }
            cap -= self.max_exposure.min(self.f0t0 / (self.flux[i].max(1e-3) * self.p.plan_q));
            self.planned[i] = true;
            n += 1;
        }
        log(&format!("season plan: {n} targets to complete"));
    }

    fn sky_model(&self, lunar: f64, alt: f64) -> f64 {
        lunar / (self.q0 * normalized_airmass(alt.max(1.0)).powf(self.airmass_exponent))
    }

    fn shaped(&self, v: f64, top_mult: f64) -> f64 {
        // Convex value of score/weight v: finishing a target beats half-doing it twice, because only
        // its best exposure counts (a partial exposure is wasted if the target is redone later).
        if self.p.kappa != 1.0 {
            top_mult * (v / top_mult).powf(self.p.kappa)
        } else {
            v
        }
    }

    fn p_success(&self, raw: f64) -> f64 {
        ((raw - self.p.req_p_lo) / (self.p.req_p_hi - self.p.req_p_lo)).max(0.0).min(1.0)
    }

    /// Return an observe action, or None when nothing useful is up.
    pub fn plan(&mut self, now: f64, night_end: f64, night_index: usize, hours: f64) -> Option<Map<String, Value>> {
        if self.p.partial_discount < 1.0 && night_index as i64 != self.plan_night {
            self.season_plan(now, night_index);
        }
        self.update_scale(hours);
        let lst = local_sidereal_deg(now, self.lon);
        let horizon = night_end.min(self.survey_end);
        let seconds_left = horizon - now;
        if seconds_left < self.min_exposure {
            return None;
        }
        let lst_later = lst + 1800.0 * SIDEREAL_DEG_PER_SECOND;
        let moon = Moon::new(now + 600.0, lst, self.lat, self.lunar_model);
        let scale = self.scale * self.p.plan_factor_safety;
        let band_scale = self.calibrated_band_scale(hours) * self.p.band_opt;
        if let Some(e_now) = self.clean_e(hours) {
            if !(self.p.weather_gate && self.all_sky_weather()) {
                let hour = hours.trunc() as i64;
                if self.e_hours.last().map_or(true, |h| h.0 != hour) {
                    self.e_hours.push((hour, night_index, Vec::new()));
                }
                self.e_hours.last_mut().unwrap().2.push(e_now);
            }
        }
        let min_up = MIN_VISIBLE_SECONDS.min(seconds_left);
        let level = self.fast_level;
        let n = self.ids.len();
        let min_up_deg = min_up * SIDEREAL_DEG_PER_SECOND;
        let pool_size = [self.p.pool, self.p.pool / 2, self.p.pool / 4, self.p.pool / 8, self.p.pool / 8][level.min(4)];
        match self.vcache.take() {
            None => self.vcache = Some((0..n).map(|i| self.value(i)).collect()),
            Some(mut cache) => {
                for &i in &self.vdirty {
                    cache[i] = self.value(i);
                }
                self.vcache = Some(cache);
            }
        }
        self.vdirty.clear();
        let night_i = night_index as i64;
        // sin(alt) = A + B cos(LST) + C sin(LST): no trigonometry per target
        let (c1, s1) = (lst.to_radians().cos(), lst.to_radians().sin());
        let (c2, s2) = ((lst + min_up_deg).to_radians().cos(), (lst + min_up_deg).to_radians().sin());
        let sin_lim = self.sin_alt_limit;
        let mut visible = vec![false; n];
        let mut any_visible = false;
        let mut still_active = Vec::with_capacity(self.active.len());
        let mut proxy: Vec<(f64, usize)> = Vec::new();
        let mut bins: HashMap<(i64, i64), f64> = HashMap::new();
        let mut bin_order: Vec<(i64, i64)> = Vec::new();
        let mut bin_best: HashMap<(i64, i64), (f64, usize)> = HashMap::new();
        {
            let vc = self.vcache.as_ref().unwrap();
            for &i in &self.active {
                let a = self.alt_a[i];
                let sin_alt = a + self.alt_b[i] * c1 + self.alt_c[i] * s1;
                if sin_alt < sin_lim || a + self.alt_b[i] * c2 + self.alt_c[i] * s2 < sin_lim {
                    still_active.push(i); // not up now (or setting soon): keep it, check its value when it rises
                    continue;
                }
                let v = vc[i];
                if v <= 0.0 {
                    continue;
                }
                still_active.push(i);
                visible[i] = true;
                any_visible = true;
                // proxy priority: planning value x a rough airmass factor x urgency (no Moon, no direction)
                let nights_left = self.last_night[i] - night_i + 1;
                proxy.push((v * sin_alt.powf(0.6) * (1.0 + self.p.urgency / (if nights_left > 1 { nights_left } else { 1 }) as f64), i));
                if self.p.n_dense > 0 {
                    // plain science still to gain here, binned on the sky at roughly one field size
                    let key = (
                        ((self.ra[i] * self.cos_dec[i]) / self.p.dense_bin_deg).floor() as i64,
                        ((self.dec[i] + 90.0) / self.p.dense_bin_deg).floor() as i64,
                    );
                    let dense_val = self.weight[i] * (1.0 - self.cur[i] / 1.2).max(0.0) * sin_alt.powf(0.6);
                    match bins.get_mut(&key) {
                        Some(total) => *total += dense_val,
                        None => {
                            bins.insert(key, dense_val);
                            bin_order.push(key);
                        }
                    }
                    if dense_val > bin_best.get(&key).map_or(0.0, |b| b.0) {
                        bin_best.insert(key, (dense_val, i));
                    }
                }
            }
        }
        self.active = still_active;
        if !any_visible {
            return None;
        }

        let durations_all: &[f64] = match level.min(3) {
            0 => &DURATIONS,
            1 => &LEVEL_DURATIONS_1,
            2 => &LEVEL_DURATIONS_2,
            _ => &LEVEL_DURATIONS_3,
        };
        let mut search = Search {
            pl: self,
            lst,
            lst_later,
            moon,
            scale,
            band_scale,
            night_i,
            base: vec![None; n],
            info: vec![None; n],
            top_mult: self.multipliers.iter().cloned().fold(f64::MIN, f64::max),
            best_rate: 0.0,
            lam: 0.0,
        };

        let mut pool: Vec<usize> = if proxy.len() > pool_size {
            let mut sorted = proxy.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            sorted.truncate(pool_size);
            sorted.into_iter().map(|(_, i)| i).collect()
        } else {
            proxy.iter().map(|(_, i)| *i).collect()
        };
        for &i in &pool {
            search.exact(i);
        }
        pool.retain(|&i| search.base[i].unwrap().is_some());
        if pool.is_empty() {
            return None;
        }

        let mut durations: Vec<f64> = durations_all.iter().cloned().filter(|&t| t <= seconds_left && t >= self.p.min_t).collect();
        if durations.is_empty() {
            durations = vec![seconds_left.trunc()];
        }
        let t_long = durations[durations.len() - 1];
        let t_mid = durations[durations.len() / 2];
        let mut lam = self.lambda_frac * self.rate_ema;
        let mut rate_decay = false;
        let mut ranked: Vec<(f64, usize)> = Vec::new();
        for &i in &pool {
            let (g1, t1) = search.quick(i, t_long);
            let (g2, t2) = search.quick(i, t_mid);
            let best_net = (g1 - lam * t1 / 16.0).max(g2 - lam * t2 / 16.0);
            if best_net > 0.0 {
                ranked.push((best_net, i));
            }
        }
        if ranked.is_empty() {
            rate_decay = true;
            lam = 0.0;
            ranked = pool.iter().map(|&i| (search.quick(i, t_long).0, i)).filter(|r| r.0 > 0.0).collect();
        }
        if ranked.is_empty() {
            drop(search);
            if rate_decay {
                self.rate_ema *= 0.9;
            }
            return None;
        }
        search.lam = lam;
        ranked.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let n_anchors = [self.p.n_anchors, 3, 1, 1][level.min(3)];
        let mut anchors: Vec<usize> = ranked.iter().take(n_anchors).map(|r| r.1).collect();
        if self.p.n_dense > 0 && level <= 1 && !bins.is_empty() {
            // also try the densest patches of remaining science: fields with no single outstanding target
            let mut by_value: Vec<(f64, (i64, i64))> = bin_order.iter().map(|k| (bins[k], *k)).collect();
            by_value.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let take = if level == 0 { self.p.n_dense } else { 2 };
            for (_, key) in by_value.into_iter().take(take) {
                let j = bin_best[&key].1; // (a KeyError in python-pro as well: the decision falls back to a wait)
                if !anchors.contains(&j) && search.exact(j).is_some() {
                    anchors.push(j);
                }
            }
        }
        let fibers: Vec<usize> = match level.min(3) {
            0 | 1 => (0..self.grid.n).collect(),
            2 => vec![5, 6, 9, 10],
            _ => vec![5],
        };
        // (net, c_alt, c_az, T, pick, total)
        let mut best: Option<(f64, f64, f64, f64, Vec<(usize, usize)>, f64)> = None;
        let mut best_near: Vec<usize> = Vec::new();
        let n_value_anchors = anchors.len().min(n_anchors);
        let radius = self.p.neighbour_radius_deg;
        for (rank, &anchor) in anchors.iter().enumerate() {
            let a = search.base[anchor].unwrap().unwrap();
            let candidates = search.pl.neighbours(search.pl.ra[anchor], search.pl.dec[anchor], radius);
            let near: Vec<usize> = candidates.into_iter().filter(|&j| visible[j] && search.exact(j).is_some()).collect();
            // density anchors mark a patch, not a target to centre: a few central placements, then refine
            let fiber_list: &[usize] = if rank < n_value_anchors { &fibers } else { &DENSE_FIBERS };
            for &fiber in fiber_list {
                let (d_north, d_east) = search.pl.grid.fiber_center(fiber);
                let (c_alt, c_az) = shift_altaz(a.alt, a.az, -d_north, -d_east);
                if !(search.pl.min_alt <= c_alt && c_alt <= 89.0) {
                    continue;
                }
                let (c_alt, c_az) = (round_to(c_alt, 4), pmod(round_to(c_az, 4), 360.0));
                if let Some(found) = search.evaluate(c_alt, c_az, &near, &durations) {
                    if best.as_ref().map_or(true, |b| found.net > b.0) {
                        best = Some((found.net, c_alt, c_az, found.t, found.pick, found.total));
                        best_near = near.clone();
                    }
                }
            }
        }
        if best.is_some() && !search.pl.p.refine_steps.is_empty() && level == 0 {
            // local search: nudge the winning pointing to catch targets near the cell edges
            let steps = search.pl.p.refine_steps.clone();
            for _ in 0..search.pl.p.refine_rounds {
                let mut improved = false;
                let (b_alt, b_az) = {
                    let b = best.as_ref().unwrap();
                    (b.1, b.2)
                };
                for &(d_north, d_east) in &steps {
                    let (c_alt, c_az) = shift_altaz(b_alt, b_az, d_north, d_east);
                    if !(search.pl.min_alt <= c_alt && c_alt <= 89.0) {
                        continue;
                    }
                    let (c_alt, c_az) = (round_to(c_alt, 4), pmod(round_to(c_az, 4), 360.0));
                    let ds: Vec<f64> = if search.pl.p.refine_fixed_t { vec![best.as_ref().unwrap().3] } else { durations.clone() };
                    if let Some(found) = search.evaluate(c_alt, c_az, &best_near, &ds) {
                        if found.net > best.as_ref().unwrap().0 + 1e-9 {
                            best = Some((found.net, c_alt, c_az, found.t, found.pick, found.total));
                            improved = true;
                        }
                    }
                }
                if !improved {
                    break;
                }
            }
        }
        let best_rate_here = search.best_rate;
        let info = std::mem::take(&mut search.info);
        drop(search);
        if rate_decay {
            self.rate_ema *= 0.9;
        }
        self.rate_ema = if self.rate_ema > 0.0 {
            (1.0 - self.p.lambda_ema) * self.rate_ema + self.p.lambda_ema * best_rate_here
        } else {
            best_rate_here
        };
        let Some((_, c_alt, c_az, t, pick, total)) = best.filter(|b| b.5 > 0.0).or(None) else {
            let hour = hours.trunc() as i64;
            if self.dbg_none_hour != Some(hour) {
                self.dbg_none_hour = Some(hour);
                log(&format!("plan none: ranked={} scale={:.3} lam={:.4}", ranked.len(), self.scale, lam));
            }
            return None;
        };
        let _ = total;
        // program: maximise expected score over the assigned targets
        let mut votes = [0.0f64; 3];
        for &(_, j) in &pick {
            let inf = info[j].unwrap();
            let model = inf.m0 + (inf.m1 - inf.m0) * (t / 3600.0).min(1.0);
            let reach = (self.flux[j] * t * model * scale / self.f0t0).min(1.0);
            votes[self.band(model * band_scale)] +=
                self.weight[j] * reach + (if self.required[j] { 0.05 * self.p.required_bonus } else { 0.0 });
        }
        let total_votes = psum(votes.iter().cloned());
        // ties: python compares (score, name), so DARK > BRIGHT > BACKUP
        let name_rank = [2, 1, 0];
        let mut program = 0;
        let mut best_key = (f64::MIN, -1);
        for k in 0..3 {
            let key = (votes[k] * self.multipliers[k] + (total_votes - votes[k]) * self.mismatch, name_rank[k]);
            if key.0 > best_key.0 || (key.0 == best_key.0 && key.1 > best_key.1) {
                best_key = key;
                program = k;
            }
        }
        self.pending = Vec::with_capacity(pick.len());
        for &(fib, j) in &pick {
            let inf = info[j].unwrap();
            let model = inf.m0 + (inf.m1 - inf.m0) * (t / 3600.0).min(1.0);
            let dir_clean = self.direction_factor(inf.alt, inf.az) >= 1.0;
            self.pending.push((
                j,
                Pred {
                    model,
                    band_model: model / 0.95,
                    alt: inf.alt,
                    az: inf.az,
                    pred: self.flux[j] * t * model * self.scale / self.f0t0,
                    dir_clean,
                    fiber: Some(fib),
                },
            ));
        }
        self.pending_program = program;
        self.pending_duration = t;
        // the mount lands at command + offset: command the desired centre minus the learned offset
        let cmd_alt = round_to((c_alt - self.offset.0).max(0.0).min(90.0), 4);
        let mut cmd_az = round_to(pmod(c_az - self.offset.1, 360.0), 4);
        if cmd_az >= 360.0 {
            cmd_az = 0.0;
        }
        self.pending_cmd = Some((cmd_alt, cmd_az));
        let mut sorted_pick = pick.clone();
        sorted_pick.sort();
        let mut assignments = Map::new();
        for (fib, j) in sorted_pick {
            assignments.insert(fib.to_string(), Value::String(self.ids[j].clone()));
        }
        let mut action = Map::new();
        action.insert("action".into(), json!("observe"));
        action.insert("pointing".into(), json!({"alt_deg": cmd_alt, "az_deg": cmd_az}));
        action.insert("assignments".into(), Value::Object(assignments));
        action.insert("duration_seconds".into(), json!(t as i64));
        action.insert("program".into(), json!(PROGRAMS[program]));
        Some(action)
    }
}

/// Per-decision caches and closures of `Planner::plan` (python-pro's exact / full / gain / quick / evaluate).
struct Search<'a> {
    pl: &'a Planner,
    lst: f64,
    lst_later: f64,
    moon: Moon,
    scale: f64,
    band_scale: f64,
    night_i: i64,
    /// None = not computed; Some(None) = not usable now.
    base: Vec<Option<Option<Base>>>,
    info: Vec<Option<Info>>,
    top_mult: f64,
    best_rate: f64,
    lam: f64,
}

impl<'a> Search<'a> {
    fn exact(&mut self, i: usize) -> Option<Base> {
        if let Some(item) = self.base[i] {
            return item;
        }
        let pl = self.pl;
        let ha = pmod(self.lst - pl.ra[i] + 180.0, 360.0) - 180.0;
        let (alt, az) = radec_to_altaz(pl.ra[i], pl.dec[i], self.lst, pl.lat);
        let dirf = pl.direction_factor(alt, az);
        if alt < pl.min_alt || dirf <= 0.0 {
            self.base[i] = Some(None);
            return None;
        }
        let h = pl.hmax[i];
        let up = if h < 180.0 { (h - ha) / SIDEREAL_DEG_PER_SECOND } else { 1e9 };
        let lunar = self.moon.lunar_factor(pl.ra[i], pl.dec[i]);
        let nights_left = (pl.last_night[i] - self.night_i + 1).max(1);
        let damp = 0.6f64.powf(pl.misses[i] as f64) * 0.8f64.powf(pl.attempts[i] as f64);
        let item = Base { alt, az, lunar, up, mult: (1.0 + pl.p.urgency / nights_left as f64) * damp * dirf };
        self.base[i] = Some(Some(item));
        Some(item)
    }

    fn full(&mut self, i: usize) -> Info {
        if let Some(item) = self.info[i] {
            return item;
        }
        let b = self.base[i].unwrap().unwrap();
        let pl = self.pl;
        let (alt2, _) = radec_to_altaz(pl.ra[i], pl.dec[i], self.lst_later, pl.lat);
        let m0 = b.lunar / (pl.q0 * normalized_airmass(b.alt.max(1.0)).powf(pl.airmass_exponent));
        let m1 = b.lunar / (pl.q0 * normalized_airmass(alt2.max(1.0)).powf(pl.airmass_exponent));
        let item = Info { m0, m1, up: b.up, mult: b.mult, alt: b.alt, az: b.az };
        self.info[i] = Some(item);
        item
    }

    fn gain(&mut self, i: usize, t: f64) -> f64 {
        let inf = self.full(i);
        if inf.up < t {
            return 0.0;
        }
        let pl = self.pl;
        let p = &pl.p;
        let model = inf.m0 + (inf.m1 - inf.m0) * (t / 3600.0).min(1.0);
        let reach = (pl.flux[i] * t * model * self.scale / pl.f0t0).min(1.0);
        let m = pl.multipliers[pl.band(model * self.band_scale)];
        let mut g = pl.weight[i] * (pl.shaped(reach * m, self.top_mult) - pl.shaped(pl.cur[i], self.top_mult)).max(0.0);
        if reach < p.partial_done && pl.planned[i] && pl.last_night[i] - self.night_i >= p.partial_nights {
            g *= p.partial_discount; // it will be completed later: this partial exposure would be wasted
        }
        if pl.required[i] && pl.factor[i] < 0.5 {
            let raw = pl.flux[i] * t * model * pl.scale / pl.f0t0 / 0.5 * pl.req_calib.get(&i).cloned().unwrap_or(1.0);
            let mut bonus = p.required_bonus * pl.p_success(raw);
            let target_best = if p.req_calendar { pl.best_future_model(i, self.night_i as usize) } else { pl.ideal_model[i] };
            if p.req_timing != 0.0 && model < p.req_timing * target_best && pl.last_night[i] - self.night_i >= p.req_timing_nights {
                bonus *= p.req_timing_discount; // a better moment for this target will come
            }
            if pl.bad_forecast && pl.last_night[i] - self.night_i >= p.req_timing_nights {
                bonus *= p.forecast_discount; // tonight is forecast bad over the whole sky
            }
            g += bonus;
        }
        if let Some(&bonus) = pl.request_bonus.get(&i) {
            let threshold = pl.request_threshold.get(&i).cloned().unwrap_or(0.5);
            let raw = pl.flux[i] * t * model * pl.scale / pl.f0t0 / threshold.max(1e-6);
            g += bonus * pl.p_success(raw);
        }
        g * inf.mult
    }

    /// gain() with the start-of-exposure sky model only (no second alt/az): for ranking.
    fn quick(&self, i: usize, t: f64) -> (f64, f64) {
        let b = self.base[i].unwrap().unwrap();
        let pl = self.pl;
        let p = &pl.p;
        let t = t.min(b.up);
        if t < pl.min_exposure {
            return (0.0, t);
        }
        let model = pl.sky_model(b.lunar, b.alt);
        let reach = (pl.flux[i] * t * model * self.scale / pl.f0t0).min(1.0);
        let mut g = pl.weight[i]
            * (pl.shaped(reach * pl.multipliers[pl.band(model * self.band_scale)], self.top_mult) - pl.shaped(pl.cur[i], self.top_mult)).max(0.0);
        if reach < p.partial_done && pl.planned[i] && pl.last_night[i] - self.night_i >= p.partial_nights {
            g *= p.partial_discount;
        }
        if pl.required[i] && pl.factor[i] < 0.5 {
            let raw = pl.flux[i] * t * model * pl.scale / pl.f0t0 / 0.5;
            g += p.required_bonus * pl.p_success(raw);
        }
        if let Some(&bonus) = pl.request_bonus.get(&i) {
            let raw = pl.flux[i] * t * model * pl.scale / pl.f0t0 / pl.request_threshold.get(&i).cloned().unwrap_or(0.5).max(1e-6);
            g += bonus * pl.p_success(raw);
        }
        (g * b.mult, t)
    }

    /// Best (net, T, pick, total) for one pointing, or None.
    fn evaluate(&mut self, c_alt: f64, c_az: f64, near: &[usize], durations: &[f64]) -> Option<Found> {
        let pl = self.pl;
        let nf = pl.grid.n;
        let mut slot = vec![usize::MAX; nf];
        let mut cells: Vec<(usize, Vec<usize>)> = Vec::new();
        for &j in near {
            let b = self.base[j].unwrap().unwrap();
            let Some((dn, de)) = tangent_offsets(b.alt, b.az, c_alt, c_az) else { continue };
            let (fib, margin) = pl.grid.classify(dn, de);
            let Some(fib) = fib else { continue };
            if margin < pl.p.edge_margin_deg * (0.5 + 1.5 * pl.misses[j] as f64) {
                continue;
            }
            if slot[fib] == usize::MAX {
                slot[fib] = cells.len();
                cells.push((fib, Vec::new()));
            }
            cells[slot[fib]].1.push(j);
        }
        if cells.is_empty() {
            return None;
        }
        let mut found: Option<Found> = None;
        for &t in durations {
            let mut total = 0.0;
            let mut pick = Vec::new();
            for (fib, js) in &cells {
                let mut top: Option<(f64, usize)> = None;
                for &j in js {
                    let g = self.gain(j, t);
                    // python max((gain, j)): larger gain, ties to the larger index
                    if top.map_or(true, |(bg, bj)| g > bg || (g == bg && j > bj)) {
                        top = Some((g, j));
                    }
                }
                let (g, j) = top.unwrap();
                if g > 0.0 {
                    total += g;
                    pick.push((*fib, j));
                }
            }
            if pick.is_empty() {
                continue;
            }
            if total / t > self.best_rate {
                self.best_rate = total / t;
            }
            let net = total - self.lam * t;
            if found.as_ref().map_or(true, |f| net > f.net) {
                found = Some(Found { net, t, pick, total });
            }
        }
        found
    }
}
