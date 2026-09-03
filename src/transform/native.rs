//! Native opencode session/message mapping (ra4).
//!
//! Semantic (recorded explicitly — see `.omo/notepads/oplire-native/ra4.md`):
//! opencode executes tools SERVER-SIDE in its agent loop. The native prompt
//! API (`POST /session/{id}/message`) accepts ONLY `text | file | agent |
//! subtask` input parts — there is NO tool-result input part and NO tools
//! parameter carrying JSON schemas (the `tools` field is just an
//! enable/disable map; the agent owns its tools server-side). Therefore:
//! - Anthropic `tool_result` input blocks are DROPPED (cannot be expressed).
//! - The request `tools` array is sent NOWHERE; entries without
//!   `input_schema` are counted as dropped schema-less tools (they could
//!   never be forwarded anyway since native has no tools param).
//! - First turn sends text (+image-as-file) parts only; completed native
//!   `tool` parts (`callID`/`tool`/`state.input`/`state.output|error`) come
//!   back paired as Anthropic `tool_use` + `tool_result` blocks in the
//!   RESPONSE so the caller sees what ran server-side.
//! - Non-empty `reasoning` parts map to Anthropic `thinking` blocks; empty
//!   reasoning text is dropped (Anthropic rejects empty thinking blocks).

use serde_json::{json, Value};

/// Native request derived from an Anthropic `/v1/messages` body.
pub struct NativeRequest {
    /// Parts for `POST /session/{id}/message` (`text | file` only).
    pub parts: Vec<Value>,
    /// System prompt for the native `system` field (optional).
    pub system: Option<String>,
    /// Native model ID (`providerID` is always `opencode`).
    pub model_id: String,
    /// Count of dropped Anthropic `tool_result` input blocks.
    pub dropped_tool_results: usize,
    /// Count of dropped schema-less entries from the `tools` array.
    pub dropped_tools: usize,
}

/// Map an Anthropic `/v1/messages` body to native session/message input.
///
/// Handles `text` / `image` / `tool_use` / `tool_result` content blocks and
/// drops schema-less tools. See module docs for the semantic.
pub fn anthropic_to_native_parts(body: &Value) -> NativeRequest {
    let model_id = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let mut parts: Vec<Value> = Vec::new();
    let mut dropped_tool_results = 0usize;
    let mut dropped_tools = 0usize;

    // System prompt: string or [{text}] -> native `system` field.
    let system = match body.get("system") {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(Value::Array(arr)) => {
            let joined = arr
                .iter()
                .filter_map(|c| c.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n");
            if joined.is_empty() {
                None
            } else {
                Some(joined)
            }
        }
        _ => None,
    };

    // Tools array is sent nowhere (native has no tools param); count
    // schema-less entries as dropped for observability.
    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        for tool in tools {
            let has_schema = tool.get("input_schema").is_some();
            if !has_schema {
                dropped_tools += 1;
            }
        }
    }

    if let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            // Assistant turns (incl. their tool_use/tool_result blocks) cannot
            // be replayed as native input — tools ran server-side already.
            if role == "assistant" {
                if let Some(arr) = msg.get("content").and_then(|v| v.as_array()) {
                    dropped_tool_results += arr
                        .iter()
                        .filter(|c| c.get("type").and_then(|t| t.as_str()) == Some("tool_result"))
                        .count();
                }
                continue;
            }
            match msg.get("content") {
                Some(Value::String(s)) => {
                    if !s.is_empty() {
                        parts.push(json!({"type": "text", "text": s}));
                    }
                }
                Some(Value::Array(arr)) => {
                    for block in arr {
                        match block.get("type").and_then(|t| t.as_str()) {
                            Some("text") => {
                                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                                    if !text.is_empty() {
                                        parts.push(json!({"type": "text", "text": text}));
                                    }
                                }
                            }
                            Some("image") => {
                                if let Some(file) = anthropic_image_to_file_part(block) {
                                    parts.push(file);
                                }
                            }
                            Some("tool_result") => {
                                // No native tool-result input part exists.
                                dropped_tool_results += 1;
                            }
                            // tool_use / thinking / redacted_thinking etc. are
                            // model outputs, not valid native input.
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
    }

    NativeRequest {
        parts,
        system,
        model_id,
        dropped_tool_results,
        dropped_tools,
    }
}

/// Map an Anthropic image block to a native `file` part.
///
/// `{type:image, source:{type:base64, media_type, data}}` becomes
/// `{type:file, mime: media_type, url: "data:<media_type>;base64,<data>"}`,
/// which satisfies `FilePartInput{type, mime, url}` (all required fields).
/// `{source:{type:url, url}}` passes the URL through with a mime fallback.
fn anthropic_image_to_file_part(block: &Value) -> Option<Value> {
    let source = block.get("source")?;
    match source.get("type").and_then(|t| t.as_str()) {
        Some("base64") => {
            let media_type = source.get("media_type").and_then(|v| v.as_str()).unwrap_or("image/png");
            let data = source.get("data").and_then(|v| v.as_str())?;
            if data.is_empty() {
                return None;
            }
            Some(json!({
                "type": "file",
                "mime": media_type,
                "url": format!("data:{};base64,{}", media_type, data)
            }))
        }
        Some("url") => {
            let url = source.get("url").and_then(|v| v.as_str())?;
            if url.is_empty() {
                return None;
            }
            // `media_type` is not part of the url-source shape; default it.
            Some(json!({
                "type": "file",
                "mime": "image/png",
                "url": url
            }))
        }
        _ => None,
    }
}

/// Map a native `session.prompt` response (`{info: AssistantMessage,
/// parts: Part[]}`) to Anthropic `/v1/messages` shape.
///
/// - `text` parts with non-empty text -> `{type:text, text}`
/// - `reasoning` parts with non-empty text -> `{type:thinking, thinking}`
///   (empty reasoning dropped — Anthropic rejects empty thinking)
/// - `tool` parts (completed: `callID`/`tool`/`state.{input,output}`,
///   error: `state.{input,error}`) -> paired `tool_use` + `tool_result`
/// - pending/running tool parts -> `tool_use` only (no result yet)
/// - `step-start` / `step-finish` / `snapshot` / `patch` / `file` / `agent`
///   parts carry no user-visible content and are dropped
/// - usage from `info.tokens{input, output}`; `stop_reason` is `tool_use`
///   when any `tool_use` block was emitted, else mapped from `info.finish`
///   (`stop`->`end_turn`, `length`->`max_tokens`).
pub fn native_response_to_anthropic(info: &Value, parts: &Value, model: &str) -> Value {
    let mut content: Vec<Value> = Vec::new();
    let mut saw_tool_use = false;

    if let Some(arr) = parts.as_array() {
        for part in arr {
            match part.get("type").and_then(|t| t.as_str()) {
                Some("text") => {
                    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                        if !text.is_empty() {
                            content.push(json!({"type": "text", "text": text}));
                        }
                    }
                }
                Some("reasoning") => {
                    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                        if !text.is_empty() {
                            content.push(json!({"type": "thinking", "thinking": text}));
                        }
                    }
                }
                Some("tool") => {
                    let call_id = part.get("callID").and_then(|v| v.as_str()).unwrap_or("");
                    let tool = part.get("tool").and_then(|v| v.as_str()).unwrap_or("");
                    if call_id.is_empty() || tool.is_empty() {
                        continue;
                    }
                    let state = part.get("state");
                    let status = state
                        .and_then(|s| s.get("status"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let input = state
                        .and_then(|s| s.get("input"))
                        .cloned()
                        .unwrap_or(json!({}));
                    content.push(json!({
                        "type": "tool_use",
                        "id": call_id,
                        "name": tool,
                        "input": input
                    }));
                    saw_tool_use = true;
                    match status {
                        "completed" => {
                            let output = state
                                .and_then(|s| s.get("output"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            content.push(json!({
                                "type": "tool_result",
                                "tool_use_id": call_id,
                                "content": output
                            }));
                        }
                        "error" => {
                            let err = state
                                .and_then(|s| s.get("error"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown error");
                            content.push(json!({
                                "type": "tool_result",
                                "tool_use_id": call_id,
                                "content": err,
                                "is_error": true
                            }));
                        }
                        // pending/running: tool_use only, no result yet.
                        _ => {}
                    }
                }
                // step-start/step-finish/snapshot/patch/file/agent/retry/
                // compaction/subtask: no user-visible content, drop.
                _ => {}
            }
        }
    }

    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }

    let finish = info.get("finish").and_then(|v| v.as_str()).unwrap_or("stop");
    let stop_reason = if saw_tool_use && finish == "stop" {
        "tool_use"
    } else {
        match finish {
            "length" => "max_tokens",
            _ => "end_turn",
        }
    };

    let tokens = info.get("tokens");
    let input_tokens = tokens
        .and_then(|t| t.get("input"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let output_tokens = tokens
        .and_then(|t| t.get("output"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    json!({
        "id": format!("msg_{}", uuid_short()),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    })
}

fn uuid_short() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{:x}{:x}", duration.as_secs(), duration.subsec_nanos())
}
