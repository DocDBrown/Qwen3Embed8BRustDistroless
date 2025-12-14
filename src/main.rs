// src/main.rs
use anyhow::{Context, Result, anyhow};
use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use half::f16;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::{DynValue, Tensor},
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, net::SocketAddr, path::PathBuf, sync::Arc};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};
use tokio::sync::Mutex;
use tower_http::{compression::CompressionLayer, cors::CorsLayer, trace::TraceLayer};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[derive(Clone)]
struct AppState {
    tokenizer: Arc<Tokenizer>,
    session: Arc<Mutex<Session>>,
    max_length: usize,
    model_name: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum InputType {
    Single(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct EmbeddingsRequest {
    #[allow(dead_code)]
    model: Option<String>,
    input: InputType,
}

#[derive(Serialize)]
struct EmbeddingData {
    object: &'static str,
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct EmbeddingsResponse {
    object: &'static str,
    data: Vec<EmbeddingData>,
    model: String,
    usage: Usage,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .with_target(false)
        .compact()
        .init();

    // Paths relative to project root (Cargo.toml)
    let model_path = PathBuf::from("models/Qwen3-Embedding-8B-onnx/onnx/model.onnx");
    let tokenizer_path = PathBuf::from("models/Qwen3-Embedding-8B-onnx/tokenizer.json");

    // Tokenizer
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("failed to load tokenizer: {e}"))
        .with_context(|| format!("path: {tokenizer_path:?}"))?;

    // You can increase this if you want longer sequences; Qwen3 supports up to 32k tokens.
    let max_length = 512usize;

    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|e| anyhow!("failed to enable truncation: {e}"))?;

    tokenizer.with_padding(Some(PaddingParams {
        strategy: tokenizers::PaddingStrategy::Fixed(max_length),
        ..Default::default()
    }));

    let tokenizer = Arc::new(tokenizer);

    // ORT v2 global init
    ort::init()
        .commit()
        .map_err(|e| anyhow!("failed to init ORT: {e}"))?;

    // Build session for the 8B ONNX model
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(&model_path)
        .map_err(|e| anyhow!("failed to load ONNX model: {e}"))
        .with_context(|| format!("path: {model_path:?}"))?;

    // Log I/O names to verify input/output wiring against the ONNX graph
    for (i, inp) in session.inputs.iter().enumerate() {
        info!("input[{i}]: {}", inp.name);
    }
    for (i, outp) in session.outputs.iter().enumerate() {
        info!("output[{i}]: {}", outp.name);
    }

    let state = AppState {
        tokenizer,
        session: Arc::new(Mutex::new(session)),
        max_length,
        model_name: "Qwen3-Embedding-8B".to_string(),
    };

    let app = Router::new()
        .route("/v1/embeddings", post(handle_embeddings))
        .layer(CorsLayer::permissive())
        .layer(CompressionLayer::new())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8981);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("listening on http://{addr}");
    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
    Ok(())
}

async fn handle_embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, (StatusCode, String)> {
    let texts: Vec<String> = match req.input {
        InputType::Single(s) => vec![s],
        InputType::Many(v) => v,
    };

    let encs = state
        .tokenizer
        .encode_batch(texts, true)
        .map_err(internal_err)?;

    let b = encs.len();
    let t = state.max_length;

    // Flat, row-major buffers
    let mut ids: Vec<i64> = vec![0; b * t];
    let mut masks: Vec<i64> = vec![0; b * t];
    let mut ttids: Vec<i64> = vec![0; b * t];

    let mut total_tokens = 0usize;

    for (i, e) in encs.iter().enumerate() {
        let ids_row = e.get_ids();
        let attn_row = e.get_attention_mask();
        let type_row = e.get_type_ids();

        total_tokens += ids_row.len();

        let take = ids_row.len().min(t);
        let base = i * t;
        for j in 0..take {
            ids[base + j] = ids_row[j] as i64;
            masks[base + j] = attn_row[j] as i64;
            ttids[base + j] = type_row[j] as i64;
        }
    }

    // Build name→DynValue map using owned buffers and simple shapes
    let shape = [b, t];

    let mut by_name: HashMap<String, DynValue> = HashMap::new();
    by_name.insert(
        "input_ids".to_string(),
        Tensor::from_array((shape, ids))
            .map_err(internal_err)?
            .into_dyn(),
    );
    by_name.insert(
        "attention_mask".to_string(),
        Tensor::from_array((shape, masks.clone()))
            .map_err(internal_err)?
            .into_dyn(),
    );
    by_name.insert(
        "token_type_ids".to_string(),
        Tensor::from_array((shape, ttids.clone()))
            .map_err(internal_err)?
            .into_dyn(),
    );

    // Build position_ids from attention_mask: cumsum over valid tokens, 0 for padding.
    let mut pos: Vec<i64> = vec![0; b * t];
    for bi in 0..b {
        let mut running: i64 = 0;
        for si in 0..t {
            let m = masks[bi * t + si];
            if m != 0 {
                pos[bi * t + si] = running;
                running += 1;
            } else {
                pos[bi * t + si] = 0;
            }
        }
    }
    by_name.insert(
        "position_ids".to_string(),
        Tensor::from_array((shape, pos))
            .map_err(internal_err)?
            .into_dyn(),
    );

    // Lock the session once, build inputs in declared order, then run.
    let mut session = state.session.lock().await;

    let mut inputs_named: Vec<(String, DynValue)> = Vec::with_capacity(session.inputs.len());
    for inp in &session.inputs {
        let lname = inp.name.to_ascii_lowercase();
        if let Some(v) = by_name.remove(&lname) {
            inputs_named.push((inp.name.clone(), v));
        } else if lname.contains("input_ids") || lname == "input" {
            if let Some(v) = by_name.remove("input_ids") {
                inputs_named.push((inp.name.clone(), v));
            } else {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("missing required input '{}'", inp.name),
                ));
            }
        } else if lname.contains("attention_mask")
            || lname.contains("position_ids")
            || lname.contains("token_type_ids")
            || lname.contains("segment")
        {
            // Zero-fill optional mask/segment inputs we did not explicitly provide
            let zeros: Vec<i64> = vec![0; b * t];
            let v = Tensor::from_array((shape, zeros))
                .map_err(internal_err)?
                .into_dyn();
            inputs_named.push((inp.name.clone(), v));
        } else {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("unsupported model input '{}'", inp.name),
            ));
        }
    }

    let outputs = session.run(inputs_named).map_err(internal_err)?;

    // Try 2D [B, D] first. Support f32 and f16. Otherwise pool [B, S, D].
    let mut embed_rows: Vec<Vec<f32>> = Vec::new();
    let mut found_any = false;

    // helper to push [B,D]
    let mut push_bd = |buf_f32: Vec<f32>, d: usize| {
        for chunk in buf_f32.chunks(d) {
            embed_rows.push(chunk.to_vec());
        }
        found_any = true;
    };

    // Scan outputs once
    for v in outputs.values() {
        if let Ok(arr) = v.try_extract_array::<f32>() {
            let shp = arr.shape();
            if shp.len() == 2 && shp[0] == b {
                let d = shp[1];
                push_bd(arr.iter().copied().collect(), d);
                break;
            }
        }
        if let Ok(arr) = v.try_extract_array::<f16>() {
            let shp = arr.shape();
            if shp.len() == 2 && shp[0] == b {
                let d = shp[1];
                let buf: Vec<f32> = arr.iter().map(|h| h.to_f32()).collect();
                push_bd(buf, d);
                break;
            }
        }
    }

    if !found_any {
        for v in outputs.values() {
            if let Ok(arr) = v.try_extract_array::<f32>() {
                let shp = arr.shape();
                if shp.len() == 3 && shp[0] == b {
                    let s = shp[1];
                    let d = shp[2];
                    let out_buf: Vec<f32> = arr.iter().copied().collect();
                    pool_masked_mean(&mut embed_rows, &out_buf, b, s, d, t, &masks);
                    found_any = true;
                    break;
                }
            }
            if let Ok(arr) = v.try_extract_array::<f16>() {
                let shp = arr.shape();
                if shp.len() == 3 && shp[0] == b {
                    let s = shp[1];
                    let d = shp[2];
                    let out_buf: Vec<f32> = arr.iter().map(|h| h.to_f32()).collect();
                    pool_masked_mean(&mut embed_rows, &out_buf, b, s, d, t, &masks);
                    found_any = true;
                    break;
                }
            }
        }
        if !found_any {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "no usable 2D or 3D tensor in outputs".to_string(),
            ));
        }
    }

    if embed_rows.len() != b {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("embedding rows {} != batch {}", embed_rows.len(), b),
        ));
    }

    let data: Vec<EmbeddingData> = embed_rows
        .into_iter()
        .enumerate()
        .map(|(i, v)| EmbeddingData {
            object: "embedding",
            embedding: v,
            index: i,
        })
        .collect();

    let resp = EmbeddingsResponse {
        object: "list",
        data,
        model: state.model_name.clone(),
        usage: Usage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    };

    Ok(Json(resp))
}

fn pool_masked_mean(
    dst: &mut Vec<Vec<f32>>,
    out_buf: &[f32],
    b: usize,
    s: usize,
    d: usize,
    t: usize,
    masks: &[i64],
) {
    for bi in 0..b {
        let mut sum = vec![0.0f32; d];
        let mut denom = 0.0f32;
        let seq_len = s.min(t);
        for si in 0..seq_len {
            if masks[bi * t + si] == 0 {
                continue;
            }
            let off = (bi * s + si) * d;
            let row = &out_buf[off..off + d];
            for (dstv, src) in sum.iter_mut().zip(row.iter()) {
                *dstv += *src;
            }
            denom += 1.0;
        }
        let denom = if denom > 0.0 { denom } else { 1.0 };
        for v in &mut sum {
            *v /= denom;
        }
        dst.push(sum);
    }
}

fn internal_err<E: std::fmt::Display>(e: E) -> (StatusCode, String) {
    error!("internal error: {e}");
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

#[cfg(test)]
mod tests_embeddings_api;

#[cfg(test)]
mod tests_server_and_pooling;
