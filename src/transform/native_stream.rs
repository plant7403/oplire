//! Native streaming SSE bridge (5uj).
//!
//! The native prompt API is UNARY (`POST /session/{id}/message` blocks until
//! the turn completes), so there is no native SSE stream to subscribe to.
//! Instead `forward_native_stream` (in `proxy::handlers`) fires
//! `POST /session/{id}/prompt_async` (204, non-blocking — same body schema as
//! the unary prompt, verified against live `/doc`) and polls
//! `GET /session/{id}/message`, diffing assistant part snapshots into an
//! Anthropic SSE event sequence.
//!
//! Mapping is VERBATIM ra4 (`native.rs`): text -> text blocks, non-empty
//! reasoning -> thinking blocks, completed/error tool parts -> tool_use
//! blocks (+tool_result pairing lives in the non-stream response; on the
//! stream the tool_use block carries the input). step-*/snapshot/patch/file/
//! agent parts are dropped. Final `stop_reason`/`usage` are derived from
//! `native_response_to_anthropic` on the terminal snapshot so stream and
//! non-stream agree for the same prompt.
//!
//! Framing mirrors the openai path (`opencode_stream_to_anthropic`):
//! `event: <name>\ndata: <json>\n\n` per event. The openai path converts
//! upstream `[DONE]` into `message_stop` and never emits a literal `[DONE]`
//! line, so this bridge also terminates with `message_stop` and no `[DONE]`.

use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

/// How many 300ms polls before giving up (≈102s, under the 120s client cap).
pub const MAX_POLLS: usize = 340;
/// Delay between message-list polls.
pub const POLL_INTERVAL_MS: u64 = 300;

#[derive(Debug)]
struct TrackedBlock {
    index: u32,
}

/// Incremental per-part emission state, keyed by native part `id` (`prt_…`).
pub struct StreamTracker {
    next_index: u32,
    blocks: HashMap<String, TrackedBlock>,
    /// Number of chars of `text` already emitted per part id.
    emitted_len: HashMap<String, usize>,
    /// Part ids whose tool_use block was already opened.
    tool_opened: HashSet<String>,
    pub saw_tool_use: bool,
}

impl StreamTracker {
    pub fn new() -> Self {
        StreamTracker {
            next_index: 0,
            blocks: HashMap::new(),
            emitted_len: HashMap::new(),
            tool_opened: HashSet::new(),
            saw_tool_use: false,
        }
    }

    /// Open block indexes in first-seen order (for the final stop sweep).
    pub fn open_indexes(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.blocks.values().map(|b| b.index).collect();
        v.sort();
        v
    }
}

impl Default for StreamTracker {
    fn default() -> Self {
        Self::new()
    }
}

fn frame(event: &str, data: &str) -> String {
    format!("event: {}\ndata: {}\n\n", event, data)
}

fn escape_json_string(s: &str) -> String {
    match serde_json::to_string(s) {
        Ok(quoted) => quoted
            .strip_prefix('"')
            .and_then(|t| t.strip_suffix('"'))
            .unwrap_or(&quoted)
            .to_string(),
        Err(_) => String::new(),
    }
}

/// `message_start` — input usage unknown until the turn ends, so 0 here;
/// real totals arrive in `message_delta` (documented 5uj choice).
pub fn message_start_event(msg_id: &str, model: &str) -> String {
    frame(
        "message_start",
        &json!({
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        })
        .to_string(),
    )
}

pub fn message_delta_event(stop_reason: &str, input_tokens: u64, output_tokens: u64) -> String {
    frame(
        "message_delta",
        &json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": null},
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}
        })
        .to_string(),
    )
}

pub fn message_stop_event() -> String {
    frame("message_stop", r#"{"type":"message_stop"}"#)
}

fn block_start_event(index: u32, block_json: &str) -> String {
    frame(
        "content_block_start",
        &format!(
            "{{\"type\":\"content_block_start\",\"index\":{},\"content_block\":{}}}",
            index, block_json
        ),
    )
}

fn block_delta_event(index: u32, delta_json: &str) -> String {
    frame(
        "content_block_delta",
        &format!(
            "{{\"type\":\"content_block_delta\",\"index\":{},\"delta\":{}}}",
            index, delta_json
        ),
    )
}

fn block_stop_event(index: u32) -> String {
    frame(
        "content_block_stop",
        &format!("{{\"type\":\"content_block_stop\",\"index\":{}}}", index),
    )
}

/// Diff one turn-parts snapshot against the tracker, returning SSE frames
/// for newly opened blocks and new text/thinking/tool deltas.
///
/// - text (non-empty): open `{type:text}` block on first sight, then
///   `text_delta` for each grown suffix.
/// - reasoning (non-empty only, ra4 rule): same with `thinking_delta`.
/// - tool completed/error with callID+tool: open `tool_use` block once and
///   emit the full input as a single `input_json_delta`. pending/running
///   parts are skipped until they settle (the terminal flush re-diffs the
///   final snapshot, so nothing is lost).
/// - everything else (step-*, snapshot, patch, file, agent, …): dropped.
pub fn diff_turn_parts(tracker: &mut StreamTracker, parts: &[Value]) -> Vec<String> {
    let mut out = Vec::new();
    for part in parts {
        let part_id = part.get("id").and_then(|v| v.as_str()).unwrap_or("");
        if part_id.is_empty() {
            continue;
        }
        match part.get("type").and_then(|t| t.as_str()) {
            Some("text") | Some("reasoning") => {
                let text = part.get("text").and_then(|t| t.as_str()).unwrap_or("");
                if text.is_empty() {
                    continue;
                }
                let is_thinking = part.get("type").and_then(|t| t.as_str()) == Some("reasoning");
                let index = match tracker.blocks.get(part_id) {
                    Some(b) => b.index,
                    None => {
                        let index = tracker.next_index;
                        tracker.next_index += 1;
                        tracker.blocks.insert(
                            part_id.to_string(),
                            TrackedBlock { index },
                        );
                        tracker.emitted_len.insert(part_id.to_string(), 0);
                        let block = if is_thinking {
                            r#"{"type":"thinking","thinking":""}"#.to_string()
                        } else {
                            r#"{"type":"text","text":""}"#.to_string()
                        };
                        out.push(block_start_event(index, &block));
                        index
                    }
                };
                let shown = tracker.emitted_len.get(part_id).copied().unwrap_or(0);
                let chars: Vec<char> = text.chars().collect();
                if chars.len() > shown {
                    let delta: String = chars[shown..].iter().collect();
                    tracker.emitted_len.insert(part_id.to_string(), chars.len());
                    let delta_json = if is_thinking {
                        format!(
                            "{{\"type\":\"thinking_delta\",\"thinking\":\"{}\"}}",
                            escape_json_string(&delta)
                        )
                    } else {
                        format!(
                            "{{\"type\":\"text_delta\",\"text\":\"{}\"}}",
                            escape_json_string(&delta)
                        )
                    };
                    out.push(block_delta_event(index, &delta_json));
                }
            }
            Some("tool") => {
                if tracker.tool_opened.contains(part_id) {
                    continue;
                }
                let status = part
                    .get("state")
                    .and_then(|s| s.get("status"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if status != "completed" && status != "error" {
                    continue;
                }
                let call_id = part.get("callID").and_then(|v| v.as_str()).unwrap_or("");
                let tool = part.get("tool").and_then(|v| v.as_str()).unwrap_or("");
                if call_id.is_empty() || tool.is_empty() {
                    continue;
                }
                let index = tracker.next_index;
                tracker.next_index += 1;
                tracker.blocks.insert(
                    part_id.to_string(),
                    TrackedBlock { index },
                );
                tracker.tool_opened.insert(part_id.to_string());
                tracker.saw_tool_use = true;
                out.push(block_start_event(
                    index,
                    &format!(
                        "{{\"type\":\"tool_use\",\"id\":\"{}\",\"name\":\"{}\",\"input\":{{}}}}",
                        escape_json_string(call_id),
                        escape_json_string(tool)
                    ),
                ));
                let input = part
                    .get("state")
                    .and_then(|s| s.get("input"))
                    .cloned()
                    .unwrap_or(json!({}));
                out.push(block_delta_event(
                    index,
                    &format!(
                        "{{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}",
                        escape_json_string(&input.to_string())
                    ),
                ));
            }
            _ => {}
        }
    }
    out
}

/// `content_block_stop` for every opened block, in index order.
pub fn stop_open_blocks(tracker: &mut StreamTracker) -> Vec<String> {
    tracker
        .open_indexes()
        .into_iter()
        .map(block_stop_event)
        .collect()
}

/// Pure turn extraction: from a `GET /session/{id}/message` array, take the
/// messages newer than `baseline` (info ids seen before `prompt_async`),
/// find our user message (first new user message; fallback: last user
/// message in the whole list), and concatenate the NEW assistant messages
/// after it in list order.
///
/// Only assistant messages newer than `baseline` are collected: on a reused
/// session the fallback anchor is the prior turn's user message, and without
/// this filter the first poll — which typically runs before `prompt_async`'s
/// user message lands (the 204 is non-blocking) — would snapshot the prior
/// turn's completed reply, report terminal immediately, and emit stale text
/// with the prior turn's usage. New-message filtering turns such premature
/// polls into empty/non-terminal so polling continues until our turn lands.
/// The fallback is kept for the server quirk where a reply lands without a
/// new user message: it then collects only the genuinely new reply.
///
/// Returns `(turn_parts, last_assistant_info, terminal)` where terminal is
/// true once the last NEW turn assistant message carries a non-null `finish`.
pub fn collect_turn_parts(
    messages: &Value,
    baseline: &HashSet<String>,
) -> (Vec<Value>, Option<Value>, bool) {
    let arr = match messages.as_array() {
        Some(a) => a,
        None => return (Vec::new(), None, false),
    };
    let is_new = |m: &Value| {
        m.get("info")
            .and_then(|i| i.get("id"))
            .and_then(|v| v.as_str())
            .map(|id| !baseline.contains(id))
            .unwrap_or(false)
    };

    let mut user_idx: Option<usize> = None;
    for (i, m) in arr.iter().enumerate() {
        if is_new(m) && m.get("info").and_then(|n| n.get("role")).and_then(|v| v.as_str()) == Some("user") {
            user_idx = Some(i);
            break;
        }
    }
    if user_idx.is_none() {
        for (i, m) in arr.iter().enumerate() {
            if m.get("info").and_then(|n| n.get("role")).and_then(|v| v.as_str()) == Some("user") {
                user_idx = Some(i);
            }
        }
    }
    let user_idx = match user_idx {
        Some(i) => i,
        None => return (Vec::new(), None, false),
    };

    let mut parts = Vec::new();
    let mut last_info: Option<Value> = None;
    for m in arr.iter().skip(user_idx + 1) {
        // Stale-history guard: only this turn's messages (newer than the
        // pre-prompt_async baseline) may contribute parts, usage, or
        // terminality. See doc comment above.
        if !is_new(m) {
            continue;
        }
        let info = match m.get("info") {
            Some(n) => n,
            None => continue,
        };
        if info.get("role").and_then(|v| v.as_str()) != Some("assistant") {
            continue;
        }
        last_info = Some(info.clone());
        if let Some(arr) = m.get("parts").and_then(|v| v.as_array()) {
            parts.extend(arr.iter().cloned());
        }
    }
    let terminal = last_info
        .as_ref()
        .and_then(|n| n.get("finish"))
        .map(|f| !f.is_null())
        .unwrap_or(false);
    (parts, last_info, terminal)
}
