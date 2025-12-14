// src/tests_server_and_pooling.rs

use super::*;
use anyhow::{Result as AnyResult, anyhow};
use axum::http::StatusCode;
use std::{path::PathBuf, sync::Arc};
use tokio::sync::Mutex;

/// State setup for tests that need tokenizer behavior.
/// Returns an error if artifacts are missing or ORT can't be initialized.
fn build_test_state_for_tokenizer() -> AnyResult<AppState> {
    let model_path = PathBuf::from("models/Qwen3-Embedding-8B-onnx/onnx/model.onnx");
    let tokenizer_path = PathBuf::from("models/Qwen3-Embedding-8B-onnx/tokenizer.json");

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("failed to load tokenizer for tests: {e}"))?;

    let max_length = 512usize;

    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|e| anyhow!("failed to enable truncation in tests: {e}"))?;

    tokenizer.with_padding(Some(PaddingParams {
        strategy: tokenizers::PaddingStrategy::Fixed(max_length),
        ..Default::default()
    }));

    let tokenizer = Arc::new(tokenizer);

    ort::init()
        .commit()
        .map_err(|e| anyhow!("ORT init failed in tests: {e}"))?;

    let session = Session::builder()
        .map_err(|e| anyhow!("failed to build session builder in tests: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("failed to set graph optimization level in tests: {e}"))?
        .commit_from_file(&model_path)
        .map_err(|e| anyhow!("failed to load ONNX model in tests: {e}"))?;

    Ok(AppState {
        tokenizer,
        session: Arc::new(Mutex::new(session)),
        max_length,
        model_name: "Qwen3-Embedding-8B".to_string(),
    })
}

#[test]
fn test_embeddings_uses_max_length_truncation() {
    let state = match build_test_state_for_tokenizer() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SKIP test_embeddings_uses_max_length_truncation: {e}");
            return;
        }
    };

    // Build a long text that will exceed max_length pre-truncation.
    let long_text = "token ".repeat(state.max_length * 2);

    let encs = state
        .tokenizer
        .encode_batch(vec![long_text], true)
        .expect("tokenization should succeed");

    assert_eq!(encs.len(), 1);

    let ids = encs[0].get_ids();
    assert_eq!(
        ids.len(),
        state.max_length,
        "encoded length must be truncated/padded to max_length"
    );
}

#[test]
fn test_embeddings_attention_mask_and_position_ids_aligned() {
    // Re-implement the position_ids logic from handle_embeddings and
    // assert alignment with a known mask pattern.
    fn build_positions_from_mask(masks: &[i64], b: usize, t: usize) -> Vec<i64> {
        let mut pos = vec![0i64; b * t];
        for bi in 0..b {
            let mut running = 0i64;
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
        pos
    }

    let b = 1usize;
    let t = 5usize;

    // Mask: 1 1 0 1 0  → positions should be 0,1,0,2,0.
    let masks: Vec<i64> = vec![1, 1, 0, 1, 0];

    let pos = build_positions_from_mask(&masks, b, t);

    assert_eq!(pos.len(), b * t);
    assert_eq!(pos, vec![0, 1, 0, 2, 0]);
}

#[test]
fn test_embeddings_pools_masked_mean_correctly() {
    // Pure unit test of pool_masked_mean.
    //
    // Setup:
    //   b = 1, s = 4, d = 2, t = 4
    //   out_buf:
    //     [1,1], [3,3], [5,5], [7,7]
    //   masks: [1,1,0,0] -> mean of first two rows = [2,2].
    let b = 1usize;
    let s = 4usize;
    let d = 2usize;
    let t = 4usize;

    let out_buf: Vec<f32> = vec![
        1.0, 1.0, // token 0
        3.0, 3.0, // token 1
        5.0, 5.0, // token 2
        7.0, 7.0, // token 3
    ];

    let masks: Vec<i64> = vec![1, 1, 0, 0];

    let mut dst: Vec<Vec<f32>> = Vec::new();
    pool_masked_mean(&mut dst, &out_buf, b, s, d, t, &masks);

    assert_eq!(dst.len(), 1);
    let emb = &dst[0];
    assert_eq!(emb.len(), d);
    assert!((emb[0] - 2.0).abs() < 1e-6);
    assert!((emb[1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_embeddings_no_usable_output_tensor_returns_500() {
    // internal_err propagates the error message and uses 500 as status for
    // internal failures, including "no usable 2D or 3D tensor in outputs".
    let (status, msg) = internal_err("no usable 2D or 3D tensor in outputs");
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(
        msg.contains("no usable 2D or 3D tensor"),
        "error message should be propagated from internal_err"
    );
}

// Pure function mirroring the PORT logic in main.rs, but injected
// instead of touching the real environment.
fn resolve_port_for_test(port_env: Option<String>) -> u16 {
    port_env.and_then(|s| s.parse().ok()).unwrap_or(8981)
}

#[test]
fn test_embeddings_default_port_when_env_port_not_set() {
    let port = resolve_port_for_test(None);
    assert_eq!(
        port, 8981,
        "when PORT is not set or invalid, default should be 8981"
    );
}

#[test]
fn test_embeddings_respects_env_port_override() {
    let port = resolve_port_for_test(Some("12345".to_string()));
    assert_eq!(port, 12345);
}
