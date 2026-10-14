//! OpenAI-compatible chat client for the pro agent. Port of `python-pro/llm_client.py`.
//!
//! Configuration (environment, or a local `.env` in the working directory that is never packed):
//!   OPENAI_API_KEY    required (KIMI_API_KEY is accepted as an alternate name)
//!   OPENAI_BASE_URL   default https://api.kimi.com/coding/v1 (Kimi Coding Plan; outside mainland China
//!                     use https://api.kimi.ai/coding/v1). On the platform this is injected automatically.
//!   OPENAI_MODEL      default k3
//!
//! Calls never block the decision loop: `submit()` starts the request on a background thread and returns a
//! handle; the agent picks the answer up with `collect()` on a later decision. k3 only accepts the default
//! temperature, so none is sent. Failed attempts (HTTP 429 / 5xx, other HTTP errors, network errors) are retried
//! with backoff (honouring `Retry-After`), bounded by the call's timeout, as in python-pro.

use serde_json::{json, Map, Value};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

pub const DEFAULT_BASE_URL: &str = "https://api.kimi.com/coding/v1";
pub const DEFAULT_MODEL: &str = "k3";

fn env_trim(name: &str) -> String {
    std::env::var(name).map(|v| v.trim().to_string()).unwrap_or_default()
}

/// OBSERVER_MODEL_DISABLED=1: the platform runs this evaluation without a model (本次不提供模型 /
/// survey26 eval start --no-model). No key is needed and no call is made: every rule default stands.
pub fn model_disabled() -> bool {
    std::env::var("OBSERVER_MODEL_DISABLED").map(|v| v == "1").unwrap_or(false)
}

pub fn api_key() -> String {
    let key = env_trim("OPENAI_API_KEY");
    if key.is_empty() {
        env_trim("KIMI_API_KEY")
    } else {
        key
    }
}

/// Fill missing environment variables from a local .env (for local runs only).
pub fn load_dotenv(path: &std::path::Path) {
    let Ok(text) = std::fs::read_to_string(path) else { return };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || !line.contains('=') {
            continue;
        }
        let (name, value) = line.split_once('=').unwrap();
        let name = name.trim();
        let value = value.trim().trim_matches('"').trim_matches('\'');
        if !name.is_empty() && !value.is_empty() && env_trim(name).is_empty() {
            std::env::set_var(name, value);
        }
    }
}

#[derive(Default)]
struct CallState {
    done: bool,
    answer: Option<Map<String, Value>>,
    error: Option<String>,
}

/// One background chat completion.
pub struct Call {
    pub tag: String,
    shared: Arc<(Mutex<CallState>, Condvar)>,
    logged: bool,
}

impl Call {
    pub fn done(&self) -> bool {
        self.shared.0.lock().map(|s| s.done).unwrap_or(true)
    }

    /// Wait up to `seconds` for the answer; true when it is done.
    pub fn wait(&self, seconds: f64) -> bool {
        let (lock, cvar) = &*self.shared;
        let deadline = Instant::now() + Duration::from_secs_f64(seconds.max(0.0));
        let mut state = match lock.lock() {
            Ok(s) => s,
            Err(_) => return true,
        };
        while !state.done {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            state = match cvar.wait_timeout(state, deadline - now) {
                Ok((s, _)) => s,
                Err(_) => return true,
            };
        }
        true
    }
}

#[derive(Clone)]
struct Endpoint {
    base_url: String,
    key: String,
    model: String,
}

/// A failed attempt: what went wrong, and the server's Retry-After (seconds) if it sent one.
enum Failure {
    Retry(String, Option<f64>),
}

impl Endpoint {
    fn request(&self, system: &str, user: &Value, timeout: f64) -> Result<Map<String, Value>, Failure> {
        let body = json!({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": serde_json::to_string(user).unwrap_or_default()}],
            "max_tokens": 2000,
        });
        let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs_f64(timeout.max(1.0))).build();
        let response = agent
            .post(&format!("{}/chat/completions", self.base_url))
            .set("Content-Type", "application/json")
            .set("Authorization", &format!("Bearer {}", self.key))
            .send_json(body);
        let data: Value = match response {
            Ok(r) => r.into_json().map_err(|e| Failure::Retry(format!("read: {}", e.kind()), None))?,
            Err(ureq::Error::Status(code, r)) => {
                // like python-pro, every failed attempt is retried; 429 / 5xx honour Retry-After
                let after = if code == 429 || code >= 500 {
                    r.header("Retry-After").and_then(|v| v.trim().parse::<f64>().ok())
                } else {
                    None
                };
                let body: String = r.into_string().unwrap_or_default().chars().filter(|c| !c.is_control()).take(120).collect();
                return Err(Failure::Retry(format!("HTTP {code}: {body}"), after));
            }
            Err(ureq::Error::Transport(t)) => return Err(Failure::Retry(format!("transport: {}", t.kind()), None)),
        };
        let text = data["choices"][0]["message"]["content"].as_str().unwrap_or("");
        // the first '{' to the last '}' (the reply may wrap the object in prose or a code fence)
        let (Some(a), Some(b)) = (text.find('{'), text.rfind('}')) else {
            return Err(Failure::Retry("no JSON object in the reply".into(), None));
        };
        if b < a {
            return Err(Failure::Retry("no JSON object in the reply".into(), None));
        }
        match serde_json::from_str::<Value>(&text[a..=b]) {
            Ok(Value::Object(map)) => Ok(map),
            _ => Err(Failure::Retry("reply is not a JSON object".into(), None)),
        }
    }
}

pub struct LlmClient {
    endpoint: Endpoint,
    pub model: String,
    call_timeout: f64,
    max_calls: usize,
    max_retries: u32,
    max_in_flight: usize,
    calls: Vec<Arc<(Mutex<CallState>, Condvar)>>,
    pub ok: usize,
    pub failed: usize,
}

impl LlmClient {
    pub fn new() -> LlmClient {
        let base = env_trim("OPENAI_BASE_URL");
        let base_url = if base.is_empty() { DEFAULT_BASE_URL.to_string() } else { base.trim_end_matches('/').to_string() };
        let model = env_trim("OPENAI_MODEL");
        let model = if model.is_empty() { DEFAULT_MODEL.to_string() } else { model };
        LlmClient {
            endpoint: Endpoint { base_url, key: api_key(), model: model.clone() },
            model,
            call_timeout: 90.0,
            max_calls: if model_disabled() { 0 } else { 1500 },
            max_retries: 3,
            max_in_flight: 4,
            calls: Vec::new(),
            ok: 0,
            failed: 0,
        }
    }

    pub fn in_flight(&self) -> usize {
        self.calls.iter().filter(|c| !c.0.lock().map(|s| s.done).unwrap_or(true)).count()
    }

    /// Start a call in the background; None when the run's limits say no.
    pub fn submit(&mut self, tag: &str, system: &'static str, user: Value, wallclock_left: f64) -> Option<Call> {
        let timeout = self.call_timeout.min(wallclock_left - 30.0);
        if self.calls.len() >= self.max_calls || timeout < 5.0 || self.in_flight() >= self.max_in_flight {
            return None;
        }
        let shared = Arc::new((Mutex::new(CallState::default()), Condvar::new()));
        let worker = shared.clone();
        let endpoint = self.endpoint.clone();
        let retries = self.max_retries;
        thread::spawn(move || {
            let started = Instant::now();
            let mut answer = None;
            let mut error = None;
            for attempt in 0..retries {
                match endpoint.request(system, &user, timeout) {
                    Ok(a) => {
                        answer = Some(a);
                        error = None;
                        break;
                    }
                    Err(Failure::Retry(e, after)) => {
                        error = Some(e);
                        if started.elapsed().as_secs_f64() > timeout {
                            break;
                        }
                        let pause = after.unwrap_or(1.0 + attempt as f64).clamp(0.5, 10.0);
                        thread::sleep(Duration::from_secs_f64(pause));
                    }
                }
            }
            let (lock, cvar) = &*worker;
            if let Ok(mut state) = lock.lock() {
                state.answer = answer;
                state.error = error;
                state.done = true;
            }
            cvar.notify_all();
        });
        self.calls.push(shared.clone());
        Some(Call { tag: tag.to_string(), shared, logged: false })
    }

    /// The parsed answer of a finished call (None while running or after a failure). Logs once.
    pub fn collect(&mut self, call: &mut Call) -> Option<Map<String, Value>> {
        let (answer, error, done) = {
            let state = call.shared.0.lock().ok()?;
            (state.answer.clone(), state.error.clone(), state.done)
        };
        if !done {
            return None;
        }
        if !call.logged {
            call.logged = true;
            if answer.is_some() {
                self.ok += 1;
            } else {
                self.failed += 1;
                // never log the key
                crate::log(&format!("llm: {} failed ({}); rules decide", call.tag, error.unwrap_or_default()));
            }
        }
        answer
    }
}
