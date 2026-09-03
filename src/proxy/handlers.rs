use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use futures::{channel::mpsc, StreamExt};
use reqwest::Client;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

use crate::config::{Backend, ProxyConfig};
use crate::transform::{anthropic_to_native_parts, anthropic_to_opencode_request, collect_turn_parts, diff_turn_parts, message_delta_event, message_start_event, message_stop_event, native_response_to_anthropic, opencode_stream_to_anthropic, opencode_response_to_anthropic, stop_open_blocks, NativeRequest, StreamTracker, MAX_POLLS, POLL_INTERVAL_MS};
use crate::warp::WarpResolver;

pub struct ProxyState {
    pub config: ProxyConfig,
    pub client: Client,
    pub warp_resolver: WarpResolver,
    /// Per-model native session cache (95w): ONE upstream session id per
    /// native model_id, reused across requests.
    ///
    /// Concurrency: `Arc<Mutex<..>>` nested INSIDE the outer
    /// `Arc<Mutex<ProxyState>>` on purpose — handlers clone this Arc under a
    /// brief outer lock, drop the outer guard, then take the inner lock ONLY
    /// for map get/insert/remove (never across network awaits). tokio's async
    /// Mutex is used because all holders are async tasks (axum handlers +
    /// spawned stream tasks); a std Mutex would block the executor.
    /// Keyed by NATIVE model_id (`native.model_id`, the opencode `modelID`),
    /// so different models never share a session. First-write races (two
    /// concurrent cold starts for the same model) resolve last-write-wins;
    /// the orphan server-side session is harmless.
    pub session_cache: SessionCache,
}

/// Per-model native session cache handle. See `ProxyState::session_cache`.
pub type SessionCache = Arc<Mutex<HashMap<String, String>>>;

/// Log-safe session prefix (first 8 chars; ids are ASCII hex/uuid).
fn short_sid(id: &str) -> String {
    id.chars().take(8).collect()
}

/// Attach the observability header proving cache behavior per response.
fn with_session_header(response: Response, reused: bool) -> Response {
    let mut response = response;
    response.headers_mut().insert(
        "x-oplire-session",
        HeaderValue::from_static(if reused { "reused" } else { "fresh" }),
    );
    response
}

pub async fn handle_models(State(state): State<Arc<Mutex<ProxyState>>>) -> impl IntoResponse {
    let state_guard = state.lock().await;
    let base_url = state_guard.config.opencode_base_url.clone();
    let api_key = state_guard.config.opencode_api_key.clone();
    drop(state_guard);

    let models_url = format!("{}/v1/models", base_url.trim_end_matches('/'));

    let mut request = reqwest::Client::new()
        .get(&models_url)
        .header("Accept", "application/json");

    if let Some(key) = &api_key {
        request = request.header("Authorization", format!("Bearer {}", key));
    }

    match request.send().await {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<Value>().await {
                Ok(upstream_models) => {
                    let transformed = transform_models_to_anthropic(&upstream_models);
                    (StatusCode::OK, Json(transformed))
                }
                Err(e) => {
                    error!("Failed to parse models response: {}", e);
                    (StatusCode::OK, Json(ProxyConfig::models_response()))
                }
            }
        }
        Ok(resp) => {
            warn!("Upstream /v1/models returned status: {}", resp.status());
            (StatusCode::OK, Json(ProxyConfig::models_response()))
        }
        Err(e) => {
            warn!("Failed to fetch models from upstream: {}", e);
            (StatusCode::OK, Json(ProxyConfig::models_response()))
        }
    }
}

fn transform_models_to_anthropic(upstream: &Value) -> Value {
    let data = upstream.get("data").and_then(|v| v.as_array());

    let models: Vec<Value> = match data {
        Some(models_array) => models_array
            .iter()
            .filter_map(|m| {
                let id = m.get("id").and_then(|v| v.as_str()).unwrap_or("");
                if id.is_empty() {
                    return None;
                }

                let display_name = match m.get("name").and_then(|v| v.as_str()) {
                    Some(name) => name.to_string(),
                    None => id.replace('-', " ")
                        .split_whitespace()
                        .map(|w| {
                            let mut chars = w.chars();
                            match chars.next() {
                                None => String::new(),
                                Some(c) => {
                                    let upper = c.to_uppercase().collect::<String>();
                                    format!("{}{}", upper, chars.as_str())
                                }
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" "),
                };

                let context_window = m
                    .get("context_window")
                    .and_then(|v| v.as_u64())
                    .or_else(|| m.get("max_tokens").and_then(|v| v.as_u64()))
                    .unwrap_or(128_000);

                let pricing_input = m
                    .get("input_price")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let pricing_output = m
                    .get("output_price")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);

                let mut model = serde_json::Map::new();
                model.insert("id".to_string(), Value::String(id.to_string()));
                model.insert("name".to_string(), Value::String(display_name.clone()));
                model.insert("type".to_string(), Value::String("model".to_string()));
                model.insert("display_name".to_string(), Value::String(display_name));
                model.insert("context_window".to_string(), Value::Number(serde_json::Number::from(context_window)));
                model.insert("input_price".to_string(), Value::Number(serde_json::Number::from_f64(pricing_input).unwrap_or(serde_json::Number::from_f64(0.0).unwrap())));
                model.insert("output_price".to_string(), Value::Number(serde_json::Number::from_f64(pricing_output).unwrap_or(serde_json::Number::from_f64(0.0).unwrap())));

                if let Some(created) = m.get("created") {
                    model.insert("created".to_string(), created.clone());
                }
                if let Some(owned_by) = m.get("owned_by") {
                    model.insert("owned_by".to_string(), owned_by.clone());
                }
                if let Some(arch) = m.get("architecture") {
                    model.insert("architecture".to_string(), arch.clone());
                }

                Some(Value::Object(model))
            })
            .collect(),
        None => {
            ProxyConfig::free_models()
                .iter()
                .map(|m| {
                    serde_json::json!({
                        "id": m.id,
                        "name": m.display_name,
                        "type": "model",
                        "display_name": m.display_name,
                        "context_window": 128000,
                        "input_price": 0.0,
                        "output_price": 0.0
                    })
                })
                .collect()
        }
    };

    serde_json::json!({
        "data": models,
        "has_more": false,
        "first_id": models.first().and_then(|m| m.get("id").and_then(|v| v.as_str())).unwrap_or(""),
        "last_id": models.last().and_then(|m| m.get("id").and_then(|v| v.as_str())).unwrap_or("")
    })
}

pub async fn handle_model_detail(
    State(state): State<Arc<Mutex<ProxyState>>>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let state_guard = state.lock().await;
    let base_url = state_guard.config.opencode_base_url.clone();
    let api_key = state_guard.config.opencode_api_key.clone();
    drop(state_guard);

    let model_url = format!("{}/v1/models/{}", base_url.trim_end_matches('/'), model_id);

    let mut request = reqwest::Client::new()
        .get(&model_url)
        .header("Accept", "application/json");

    if let Some(key) = &api_key {
        request = request.header("Authorization", format!("Bearer {}", key));
    }

    match request.send().await {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<Value>().await {
                Ok(model) => {
                    let mut transformed = serde_json::Map::new();
                    if let Some(id) = model.get("id").and_then(|v| v.as_str()) {
                        transformed.insert("id".to_string(), Value::String(id.to_string()));
                    }
                    if let Some(name) = model.get("name").and_then(|v| v.as_str()) {
                        transformed.insert("name".to_string(), Value::String(name.to_string()));
                        transformed.insert("display_name".to_string(), Value::String(name.to_string()));
                    }
                    transformed.insert("type".to_string(), Value::String("model".to_string()));

                    (StatusCode::OK, Json(Value::Object(transformed))).into_response()
                }
                Err(e) => {
                    error!("Failed to parse model detail: {}", e);
                    StatusCode::NOT_FOUND.into_response()
                }
            }
        }
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

pub async fn handle_messages(
    State(state): State<Arc<Mutex<ProxyState>>>,
    _headers: HeaderMap,
    body: Bytes,
) -> Response {
    let request_body = match serde_json::from_slice::<Value>(&body) {
        Ok(v) => v,
        Err(e) => {
            error!("Failed to parse request body: {}", e);
            return error_response(&format!("Invalid JSON: {}", e));
        }
    };

    let is_stream = request_body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let model = request_body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let opencode_body = anthropic_to_opencode_request(&request_body);

    let state_guard = state.lock().await;
    let base_url = state_guard.config.opencode_base_url.clone();
    let api_key = state_guard.config.opencode_api_key.clone();
    let max_retries = state_guard.config.max_retries;
    let reset_delay = state_guard.config.warp_reset_delay_ms;
    let backend = state_guard.config.backend;
    let session_cache = state_guard.session_cache.clone();
    drop(state_guard);

    // Native streams via prompt_async + message-list polling (5uj).
    let mut retry_count = 0;

    loop {
        let result = match backend {
            Backend::Native => {
                if is_stream {
                    forward_native_stream(&base_url, &api_key, &request_body, &model, &session_cache).await
                } else {
                    forward_native(&base_url, &api_key, &request_body, &model, &session_cache).await
                }
            }
            Backend::Openai => {
                if is_stream {
                    forward_streaming(&base_url, &api_key, &opencode_body, &model).await
                } else {
                    forward_non_streaming(&base_url, &api_key, &opencode_body, &model).await
                }
            }
        };

        match result {
            Ok(response) => return response,
            Err(ProxyError::RateLimited) => {
                retry_count += 1;
                if retry_count > max_retries {
                    return error_response("Rate limit exceeded after WARP resets");
                }

                info!(
                    "429 received, attempting WARP reset (attempt {}/{})",
                    retry_count, max_retries
                );

                let resolver = WarpResolver::new(max_retries, reset_delay);
                if !resolver.handle_429(retry_count - 1).await {
                    return error_response("WARP reset failed, rate limit still active");
                }
            }
            Err(ProxyError::RequestFailed(msg)) => {
                error!("Upstream request failed: {}", msg);
                return error_response(&format!("Upstream error: {}", msg));
            }
        }
    }
}

async fn forward_streaming(
    base_url: &str,
    api_key: &Option<String>,
    body: &Value,
    model: &str,
) -> Result<Response, ProxyError> {
    let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));

    let mut request = reqwest::Client::new()
        .post(&url)
        .header("Content-Type", "application/json")
        .header("Accept", "text/event-stream")
        .json(body);

    if let Some(key) = api_key {
        request = request.header("Authorization", format!("Bearer {}", key));
    }

    let response = request
        .send()
        .await
        .map_err(|e| ProxyError::RequestFailed(e.to_string()))?;

    if response.status() == 429 {
        return Err(ProxyError::RateLimited);
    }

    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(ProxyError::RequestFailed(format!(
            "Upstream {} : {}",
            status, body
        )));
    }

    let model_owned = model.to_string();
    let stream = response
        .bytes_stream()
        .map(move |chunk| {
            let chunk = chunk.map_err(std::io::Error::other)?;
            let lines = String::from_utf8_lossy(&chunk);
            let mut output = String::new();

            for line in lines.lines() {
                if let Some(transformed) = opencode_stream_to_anthropic(line, &model_owned) {
                    output.push_str(&transformed);
                }
            }

            if output.is_empty() {
                Ok::<Bytes, std::io::Error>(Bytes::from(lines.into_owned()))
            } else {
                Ok::<Bytes, std::io::Error>(Bytes::from(output))
            }
        });

    let body = Body::from_stream(stream);

    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        "Content-Type",
        HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    response.headers_mut().insert(
        "Cache-Control",
        HeaderValue::from_static("no-cache"),
    );
    response
        .headers_mut()
        .insert("Connection", HeaderValue::from_static("keep-alive"));
    response
        .headers_mut()
        .insert("X-Accel-Buffering", HeaderValue::from_static("no"));

    Ok(response)
}

/// Reset table (95w) — how native upstream HTTP statuses are classified:
///
/// | status        | classification | session action              |
/// |---------------|----------------|-------------------------------|
/// | 429           | RateLimited    | KEEP session; outer WARP loop |
/// | other 4xx     | Stale          | DROP cached id, recreate ONCE |
/// | 5xx           | Failed         | KEEP session (server-side     |
/// |               |                | state likely intact)          |
/// | network error | Failed         | KEEP session (no signal the  |
/// |               |                | server dropped it)            |
///
/// Rationale for keeping on 5xx/network: a 4xx names THIS session/message as
/// bad (unknown session, bad member id, validation tied to session state);
/// a 5xx/network error says the server stumbled, not that our session id is
/// invalid — dropping would churn sessions without cause.
#[derive(Debug)]
enum NativeAttemptError {
    RateLimited,
    Stale(String),
    Failed(String),
}

/// Classify a native upstream response WITHOUT consuming success bodies:
/// success passes the response through, everything else becomes a typed
/// error per the reset table above.
enum NativeOutcome {
    Ok(reqwest::Response),
    Err(NativeAttemptError),
}

async fn classify_native_response(response: reqwest::Response) -> NativeOutcome {
    if response.status().is_success() {
        return NativeOutcome::Ok(response);
    }
    let status = response.status();
    let body = response
        .text()
        .await
        .unwrap_or_else(|_| "Unknown error".to_string());
    let msg = format!("Native upstream {} : {}", status, body);
    let err = if status.as_u16() == 429 {
        NativeAttemptError::RateLimited
    } else if status.is_client_error() {
        NativeAttemptError::Stale(msg)
    } else {
        NativeAttemptError::Failed(msg)
    };
    NativeOutcome::Err(err)
}

async fn create_native_session(
    root: &str,
    api_key: &Option<String>,
) -> Result<String, ProxyError> {
    let session_url = format!("{}/session", root);
    let mut session_req = reqwest::Client::new()
        .post(&session_url)
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({}));
    if let Some(key) = api_key {
        session_req = session_req.header("Authorization", format!("Bearer {}", key));
    }
    let session_resp = session_req
        .send()
        .await
        .map_err(|e| ProxyError::RequestFailed(e.to_string()))?;
    let session_resp = match classify_native_response(session_resp).await {
        NativeOutcome::Ok(response) => response,
        NativeOutcome::Err(NativeAttemptError::RateLimited) => {
            return Err(ProxyError::RateLimited);
        }
        NativeOutcome::Err(NativeAttemptError::Stale(msg))
        | NativeOutcome::Err(NativeAttemptError::Failed(msg)) => {
            return Err(ProxyError::RequestFailed(msg));
        }
    };
    let session: Value = session_resp
        .json()
        .await
        .map_err(|e| ProxyError::RequestFailed(e.to_string()))?;
    session
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| {
            ProxyError::RequestFailed("native session response missing id".to_string())
        })
}

/// Drop the cached session id ONLY if it is still the value we used
/// (avoids clobbering a fresher id stored concurrently). Lock held for the
/// map op only, never across network I/O.
async fn drop_cached_session(session_cache: &SessionCache, cache_key: &str, session_id: &str) {
    let mut guard = session_cache.lock().await;
    if guard.get(cache_key).map(|s| s.as_str()) == Some(session_id) {
        guard.remove(cache_key);
    }
}
async fn forward_native(
    base_url: &str,
    api_key: &Option<String>,
    body: &Value,
    model: &str,
    session_cache: &SessionCache,
) -> Result<Response, ProxyError> {
    if !is_loopback_url(base_url) {
        return Err(ProxyError::RequestFailed(format!(
            "refusing non-loopback native upstream: {}",
            base_url
        )));
    }

    let native = anthropic_to_native_parts(body);
    if native.dropped_tool_results > 0 || native.dropped_tools > 0 {
        info!(
            "native mapping dropped {} tool_result block(s) and {} schema-less tool(s): tools execute server-side, no native input exists for them",
            native.dropped_tool_results, native.dropped_tools
        );
    }
    if native.parts.is_empty() {
        return Err(ProxyError::RequestFailed(
            "no mappable native input parts (only text/image content is supported)".to_string(),
        ));
    }

    let root = base_url.trim_end_matches('/');
    let cache_key = native.model_id.clone();

    let cached: Option<String> = session_cache.lock().await.get(&cache_key).cloned();
    let (mut session_id, mut reused) = match cached {
        Some(id) => {
            info!(
                "native session reuse model={} session={}…",
                cache_key,
                short_sid(&id)
            );
            (id, true)
        }
        None => {
            let id = create_native_session(root, api_key).await?;
            info!(
                "native session create model={} session={}…",
                cache_key,
                short_sid(&id)
            );
            session_cache
                .lock()
                .await
                .insert(cache_key.clone(), id.clone());
            (id, false)
        }
    };

    for _ in 0..2 {
        match forward_native_with_session(root, api_key, model, &native, &session_id).await {
            Ok(response) => return Ok(with_session_header(response, reused)),
            Err(NativeAttemptError::RateLimited) => return Err(ProxyError::RateLimited),
            Err(NativeAttemptError::Failed(msg)) => {
                return Err(ProxyError::RequestFailed(msg));
            }
            Err(NativeAttemptError::Stale(msg)) => {
                if !reused {
                    return Err(ProxyError::RequestFailed(msg));
                }
                info!(
                    "native session reset model={} session={}… ({} — retrying once fresh)",
                    cache_key,
                    short_sid(&session_id),
                    msg
                );
                drop_cached_session(session_cache, &cache_key, &session_id).await;
                session_id = create_native_session(root, api_key).await?;
                info!(
                    "native session create model={} session={}… (post-reset)",
                    cache_key,
                    short_sid(&session_id)
                );
                session_cache
                    .lock()
                    .await
                    .insert(cache_key.clone(), session_id.clone());
                reused = false;
            }
        }
    }

    Err(ProxyError::RequestFailed(
        "native session retry exhausted".to_string(),
    ))
}

async fn forward_native_with_session(
    root: &str,
    api_key: &Option<String>,
    model: &str,
    native: &NativeRequest,
    session_id: &str,
) -> Result<Response, NativeAttemptError> {
    let mut message = serde_json::json!({
        "model": {"providerID": "opencode", "modelID": native.model_id},
        "parts": native.parts
    });
    if let Some(system) = native.system.clone() {
        if let Some(obj) = message.as_object_mut() {
            obj.insert("system".to_string(), Value::String(system));
        }
    }

    let message_url = format!("{}/session/{}/message", root, session_id);
    let mut message_req = reqwest::Client::new()
        .post(&message_url)
        .header("Content-Type", "application/json")
        .json(&message);
    if let Some(key) = api_key {
        message_req = message_req.header("Authorization", format!("Bearer {}", key));
    }
    let message_resp = message_req
        .send()
        .await
        .map_err(|e| NativeAttemptError::Failed(e.to_string()))?;
    let message_resp = match classify_native_response(message_resp).await {
        NativeOutcome::Ok(response) => response,
        NativeOutcome::Err(err) => return Err(err),
    };
    let prompt_resp: Value = message_resp
        .json()
        .await
        .map_err(|e| NativeAttemptError::Failed(e.to_string()))?;
    let mut info = prompt_resp
        .get("info")
        .cloned()
        .unwrap_or(Value::Null);
    let mut parts = prompt_resp
        .get("parts")
        .cloned()
        .unwrap_or(Value::Null);

    // Tool executions live in intermediate assistant messages, NOT in the
    // prompt response (which carries only the final message). Merge the
    // turn's assistant parts via the message list so completed tool parts
    // surface as paired tool_use + tool_result blocks.
    match turn_parts_via_history_attempt(root, api_key, session_id, &info).await {
        Ok(Some((turn_info, turn_parts))) => {
            info = turn_info;
            parts = turn_parts;
        }
        Ok(None) => {}
        Err(err) => return Err(err),
    }

    let anthropic_response = native_response_to_anthropic(&info, &parts, model);

    let mut headers = HeaderMap::new();
    headers.insert(
        "Content-Type",
        HeaderValue::from_static("application/json"),
    );
    headers.insert(
        "x-request-id",
        HeaderValue::from_str(&format!("req_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0))).unwrap_or(HeaderValue::from_static("unknown")),
    );

    Ok((StatusCode::OK, headers, Json(anthropic_response)).into_response())
}

fn is_loopback_url(base_url: &str) -> bool {
    match base_url.parse::<url::Url>() {
        Ok(url) => match url.host_str() {
            Some("localhost") | Some("127.0.0.1") | Some("::1") => true,
            Some(host) => host.starts_with("127."),
            None => false,
        },
        Err(_) => false,
    }
}

async fn forward_native_stream(
    base_url: &str,
    api_key: &Option<String>,
    body: &Value,
    model: &str,
    session_cache: &SessionCache,
) -> Result<Response, ProxyError> {
    if !is_loopback_url(base_url) {
        return Err(ProxyError::RequestFailed(format!(
            "refusing non-loopback native upstream: {}",
            base_url
        )));
    }

    let native = anthropic_to_native_parts(body);
    if native.dropped_tool_results > 0 || native.dropped_tools > 0 {
        info!(
            "native mapping dropped {} tool_result block(s) and {} schema-less tool(s): tools execute server-side, no native input exists for them",
            native.dropped_tool_results, native.dropped_tools
        );
    }
    if native.parts.is_empty() {
        return Err(ProxyError::RequestFailed(
            "no mappable native input parts (only text/image content is supported)".to_string(),
        ));
    }

    let root = base_url.trim_end_matches('/').to_string();
    let client = reqwest::Client::new();
    let cache_key = native.model_id.clone();

    let cached: Option<String> = session_cache.lock().await.get(&cache_key).cloned();
    let (mut session_id, mut reused) = match cached {
        Some(id) => {
            info!(
                "native session reuse model={} session={}… (stream)",
                cache_key,
                short_sid(&id)
            );
            (id, true)
        }
        None => {
            let id = create_native_session(&root, api_key).await?;
            info!(
                "native session create model={} session={}… (stream)",
                cache_key,
                short_sid(&id)
            );
            session_cache
                .lock()
                .await
                .insert(cache_key.clone(), id.clone());
            (id, false)
        }
    };

    let mut baseline = HashSet::new();
    for _ in 0..2 {
        match start_native_stream_attempt(&client, api_key, &root, &native, &session_id).await {
            Ok(ids) => {
                baseline = ids;
                break;
            }
            Err(NativeAttemptError::RateLimited) => return Err(ProxyError::RateLimited),
            Err(NativeAttemptError::Failed(msg)) => {
                return Err(ProxyError::RequestFailed(msg));
            }
            Err(NativeAttemptError::Stale(msg)) => {
                if !reused {
                    return Err(ProxyError::RequestFailed(msg));
                }
                info!(
                    "native session reset model={} session={}… (stream: {} — retrying once fresh)",
                    cache_key,
                    short_sid(&session_id),
                    msg
                );
                drop_cached_session(session_cache, &cache_key, &session_id).await;
                session_id = create_native_session(&root, api_key).await?;
                info!(
                    "native session create model={} session={}… (stream post-reset)",
                    cache_key,
                    short_sid(&session_id)
                );
                session_cache
                    .lock()
                    .await
                    .insert(cache_key.clone(), session_id.clone());
                reused = false;
            }
        }
    }

    let (tx, rx) = mpsc::unbounded::<Bytes>();
    let task_client = client;
    let task_key = api_key.clone();
    let task_model = model.to_string();
    tokio::spawn(async move {
        run_native_stream_task(
            &task_client,
            &root,
            &task_key,
            &session_id,
            &baseline,
            &task_model,
            tx,
        )
        .await;
    });

    let stream = rx.map(|bytes| Ok::<Bytes, std::io::Error>(bytes));
    let body = Body::from_stream(stream);

    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        "Content-Type",
        HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    response.headers_mut().insert(
        "Cache-Control",
        HeaderValue::from_static("no-cache"),
    );
    response
        .headers_mut()
        .insert("Connection", HeaderValue::from_static("keep-alive"));
    response
        .headers_mut()
        .insert("X-Accel-Buffering", HeaderValue::from_static("no"));

    Ok(with_session_header(response, reused))
}

/// Baseline fetch + prompt_async kick for ONE stream attempt on an existing
/// session. Returns the baseline message-id set on success.
async fn start_native_stream_attempt(
    client: &reqwest::Client,
    api_key: &Option<String>,
    root: &str,
    native: &NativeRequest,
    session_id: &str,
) -> Result<HashSet<String>, NativeAttemptError> {
    let baseline_messages =
        fetch_native_message_list_attempt(client, api_key, root, session_id).await?;
    let baseline = message_id_set(&baseline_messages);

    let mut message = serde_json::json!({
        "model": {"providerID": "opencode", "modelID": native.model_id},
        "parts": native.parts
    });
    if let Some(system) = native.system.clone() {
        if let Some(obj) = message.as_object_mut() {
            obj.insert("system".to_string(), Value::String(system));
        }
    }

    let mut async_req = client
        .post(format!("{}/session/{}/prompt_async", root, session_id))
        .header("Content-Type", "application/json")
        .json(&message);
    if let Some(key) = api_key {
        async_req = async_req.header("Authorization", format!("Bearer {}", key));
    }
    let async_resp = async_req
        .send()
        .await
        .map_err(|e| NativeAttemptError::Failed(e.to_string()))?;
    match classify_native_response(async_resp).await {
        NativeOutcome::Ok(_) => Ok(baseline),
        NativeOutcome::Err(err) => Err(err),
    }
}

async fn fetch_native_message_list_attempt(
    client: &reqwest::Client,
    api_key: &Option<String>,
    root: &str,
    session_id: &str,
) -> Result<Value, NativeAttemptError> {
    let mut history_req = client
        .get(format!("{}/session/{}/message", root, session_id))
        .header("Accept", "application/json");
    if let Some(key) = api_key {
        history_req = history_req.header("Authorization", format!("Bearer {}", key));
    }
    let history_resp = history_req
        .send()
        .await
        .map_err(|e| NativeAttemptError::Failed(e.to_string()))?;
    let history_resp = match classify_native_response(history_resp).await {
        NativeOutcome::Ok(response) => response,
        NativeOutcome::Err(err) => return Err(err),
    };
    history_resp
        .json()
        .await
        .map_err(|e| NativeAttemptError::Failed(e.to_string()))
}

async fn fetch_native_message_list(
    client: &reqwest::Client,
    api_key: &Option<String>,
    root: &str,
    session_id: &str,
) -> Result<Value, ProxyError> {
    match fetch_native_message_list_attempt(client, api_key, root, session_id).await {
        Ok(value) => Ok(value),
        Err(NativeAttemptError::RateLimited) => Err(ProxyError::RateLimited),
        Err(NativeAttemptError::Stale(msg)) | Err(NativeAttemptError::Failed(msg)) => {
            Err(ProxyError::RequestFailed(msg))
        }
    }
}

fn message_id_set(messages: &Value) -> HashSet<String> {
    messages
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|m| {
                    m.get("info")
                        .and_then(|n| n.get("id"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                })
                .collect()
        })
        .unwrap_or_default()
}

async fn run_native_stream_task(
    client: &reqwest::Client,
    root: &str,
    api_key: &Option<String>,
    session_id: &str,
    baseline: &HashSet<String>,
    model: &str,
    tx: mpsc::UnboundedSender<Bytes>,
) {
    let since_epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| (d.as_secs(), d.subsec_nanos()))
        .unwrap_or((0, 0));
    let msg_id = format!("msg_{:x}{:x}", since_epoch.0, since_epoch.1);

    let send = |s: String| tx.unbounded_send(Bytes::from(s)).is_ok();
    if !send(message_start_event(&msg_id, model)) {
        return;
    }

    let mut tracker = StreamTracker::new();
    let mut last_info: Option<Value> = None;
    let mut last_parts: Vec<Value> = Vec::new();

    for _ in 0..MAX_POLLS {
        match fetch_native_message_list(client, api_key, root, session_id).await {
            Ok(messages) => {
                let (parts, info, done) = collect_turn_parts(&messages, baseline);
                for ev in diff_turn_parts(&mut tracker, &parts) {
                    if !send(ev) {
                        return;
                    }
                }
                if info.is_some() {
                    last_info = info;
                }
                last_parts = parts;
                if done {
                    break;
                }
            }
            Err(_) => {}
        }
        tokio::time::sleep(std::time::Duration::from_millis(POLL_INTERVAL_MS)).await;
    }

    if let Ok(messages) = fetch_native_message_list(client, api_key, root, session_id).await {
        let (parts, info, _) = collect_turn_parts(&messages, baseline);
        for ev in diff_turn_parts(&mut tracker, &parts) {
            if !send(ev) {
                return;
            }
        }
        if info.is_some() {
            last_info = info;
        }
        last_parts = parts;
    }

    for ev in stop_open_blocks(&mut tracker) {
        if !send(ev) {
            return;
        }
    }

    let canonical = native_response_to_anthropic(
        last_info.as_ref().unwrap_or(&Value::Null),
        &Value::Array(last_parts),
        model,
    );
    let stop_reason = canonical
        .get("stop_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("end_turn");
    let usage = canonical.get("usage");
    let input_tokens = usage
        .and_then(|u| u.get("input_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let output_tokens = usage
        .and_then(|u| u.get("output_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let _ = send(message_delta_event(stop_reason, input_tokens, output_tokens));
    let _ = send(message_stop_event());
}

async fn turn_parts_via_history_attempt(
    root: &str,
    api_key: &Option<String>,
    session_id: &str,
    prompt_info: &Value,
) -> Result<Option<(Value, Value)>, NativeAttemptError> {
    let user_msg_id = match prompt_info.get("parentID").and_then(|v| v.as_str()) {
        Some(id) => id,
        None => return Ok(None),
    };
    let history_url = format!("{}/session/{}/message", root, session_id);
    let mut history_req = reqwest::Client::new()
        .get(&history_url)
        .header("Accept", "application/json");
    if let Some(key) = api_key {
        history_req = history_req.header("Authorization", format!("Bearer {}", key));
    }
    let history_resp = history_req
        .send()
        .await
        .map_err(|e| NativeAttemptError::Failed(e.to_string()))?;
    if !history_resp.status().is_success() {
        let status = history_resp.status();
        // 5xx on the history read is non-fatal: keep the session and fall
        // back to the prompt response (reset table: 5xx keeps session).
        if status.is_server_error() {
            return Ok(None);
        }
        let body = history_resp
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        let msg = format!("Native upstream {} : {}", status, body);
        if status.as_u16() == 429 {
            return Err(NativeAttemptError::RateLimited);
        } else if status.is_client_error() {
            return Err(NativeAttemptError::Stale(msg));
        } else {
            return Err(NativeAttemptError::Failed(msg));
        }
    }
    let messages: Value = history_resp
        .json()
        .await
        .map_err(|e| NativeAttemptError::Failed(e.to_string()))?;

    let messages = match messages.as_array() {
        Some(arr) => arr,
        None => return Ok(None),
    };

    let mut seen_user_turn = false;
    let mut turn_parts: Vec<Value> = Vec::new();
    let mut last_info = prompt_info.clone();
    for msg in messages {
        let msg_info = match msg.get("info") {
            Some(info) => info,
            None => continue,
        };
        let msg_id = msg_info.get("id").and_then(|v| v.as_str()).unwrap_or("");
        if !seen_user_turn {
            if msg_id == user_msg_id {
                seen_user_turn = true;
            }
            continue;
        }
        if msg_info.get("role").and_then(|v| v.as_str()) != Some("assistant") {
            continue;
        }
        last_info = msg_info.clone();
        if let Some(arr) = msg.get("parts").and_then(|v| v.as_array()) {
            turn_parts.extend(arr.iter().cloned());
        }
    }
    if !seen_user_turn || turn_parts.is_empty() {
        return Ok(None);
    }
    Ok(Some((last_info, Value::Array(turn_parts))))
}

async fn forward_non_streaming(
    base_url: &str,
    api_key: &Option<String>,
    body: &Value,
    model: &str,
) -> Result<Response, ProxyError> {
    let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));

    let mut request = reqwest::Client::new()
        .post(&url)
        .header("Content-Type", "application/json")
        .json(body);

    if let Some(key) = api_key {
        request = request.header("Authorization", format!("Bearer {}", key));
    }

    let response = request
        .send()
        .await
        .map_err(|e| ProxyError::RequestFailed(e.to_string()))?;

    if response.status() == 429 {
        return Err(ProxyError::RateLimited);
    }

    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(ProxyError::RequestFailed(format!(
            "Upstream {} : {}",
            status, body
        )));
    }

    let opencode_response = response
        .json::<Value>()
        .await
        .map_err(|e| ProxyError::RequestFailed(e.to_string()))?;

    let anthropic_response = opencode_response_to_anthropic(&opencode_response, model);

    let mut headers = HeaderMap::new();
    headers.insert(
        "Content-Type",
        HeaderValue::from_static("application/json"),
    );
    headers.insert(
        "x-request-id",
        HeaderValue::from_str(&format!("req_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0))).unwrap_or(HeaderValue::from_static("unknown")),
    );

    Ok((StatusCode::OK, headers, Json(anthropic_response)).into_response())
}

fn error_response(message: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "api_error",
            "code": "internal_error"
        }
    });
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(body),
    )
        .into_response()
}

#[derive(Debug)]
enum ProxyError {
    RateLimited,
    RequestFailed(String),
}
