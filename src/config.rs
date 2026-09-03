/// Backend used by the proxy bridge (stored/plumbed only; not branched on yet).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// Native opencode prompt/session API (default).
    #[default]
    Native,
    /// Legacy OpenAI-compatible chat completions path.
    Openai,
}

/// Proxy configuration for the Anthropic ↔ OpenCode Zen bridge.
#[derive(Debug, Clone)]
pub struct ProxyConfig {
    /// Address to listen on (default: 127.0.0.1:8080)
    pub listen_addr: String,
    /// OpenCode Zen API base URL (default: http://localhost:3000)
    pub opencode_base_url: String,
    /// API key for OpenCode Zen (if required)
    pub opencode_api_key: Option<String>,
    /// Maximum retry attempts after WARP reset
    pub max_retries: u32,
    /// Delay between WARP reset steps (milliseconds)
    pub warp_reset_delay_ms: u64,
    /// Backend used by the proxy bridge (default: native)
    pub backend: Backend,
}

/// A free model available on OpenCode Zen.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FreeModel {
    /// Model identifier used by OpenCode Zen
    pub id: String,
    /// Display name shown in Claude Code
    pub display_name: String,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            listen_addr: "127.0.0.1:8080".to_string(),
            opencode_base_url: "http://localhost:3000".to_string(),
            opencode_api_key: None,
            max_retries: 3,
            warp_reset_delay_ms: 5000,
            backend: Backend::default(),
        }
    }
}

impl ProxyConfig {
    /// Returns the list of free models mapped to Anthropic schema.
    pub fn free_models() -> Vec<FreeModel> {
        vec![
            FreeModel {
                id: "big-pickle".to_string(),
                display_name: "Big Pickle".to_string(),
            },
            FreeModel {
                id: "ling-3.0-flash-fin-free".to_string(),
                display_name: "Ling 3.0 Flash Fin Free".to_string(),
            },
            FreeModel {
                id: "mimo-v2.5-free".to_string(),
                display_name: "Mimo V2.5 Free".to_string(),
            },
            FreeModel {
                id: "muse-spark-1.2-contributor-free".to_string(),
                display_name: "Muse Spark 1.2 Contributor Free".to_string(),
            },
            FreeModel {
                id: "muse-spark-1.3-contributor-free".to_string(),
                display_name: "Muse Spark 1.3 Contributor Free".to_string(),
            },
            FreeModel {
                id: "nemotron-3-ultra-free".to_string(),
                display_name: "Nemotron 3 Ultra Free".to_string(),
            },
            FreeModel {
                id: "nemotron-3.5-lightning-free".to_string(),
                display_name: "Nemotron 3.5 Lightning Free".to_string(),
            },
        ]
    }

    /// Build Anthropic-compatible models response JSON.
    pub fn models_response() -> serde_json::Value {
        let models: Vec<serde_json::Value> = Self::free_models()
            .iter()
            .map(|m| {
                serde_json::json!({
                    "id": m.id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "opencode-zen"
                })
            })
            .collect();

        serde_json::json!({
            "object": "list",
            "data": models
        })
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AppConfig {
    pub listen: String,
    pub upstream: String,
    pub max_retries: u32,
    pub warp_delay: u64,
    #[serde(default)]
    pub backend: Backend,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:8080".to_string(),
            upstream: "http://localhost:3000".to_string(),
            max_retries: 3,
            warp_delay: 5000,
            backend: Backend::default(),
        }
    }
}
