# ---------- builder ----------
FROM rust:1.90-bookworm AS builder
ENV CARGO_TERM_COLOR=always RUSTFLAGS="-C strip=symbols"
WORKDIR /src

# Prebuild deps
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src && echo "fn main(){}" > src/main.rs && \
    cargo build --release && \
    rm -rf target/release/deps/qwen_embed_8_rs* && rm -rf src

# Build real binary
COPY src ./src
RUN cargo build --release

# ---------- fetch onnxruntime ----------
FROM debian:bookworm-slim AS ort
ARG ORT_VERSION=1.23.2
WORKDIR /opt/onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl tar && \
    curl -fsSL -o onnxruntime.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" && \
    tar -xzf onnxruntime.tgz --strip-components=1 && rm -f onnxruntime.tgz

# ---------- runtime dep ----------
FROM debian:bookworm-slim AS syslibs
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# ---------- model (robust downloader) ----------
FROM debian:bookworm-slim AS model
WORKDIR /download
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates aria2 && \
    rm -rf /var/lib/apt/lists/*

# Hugging Face 8B ONNX with external data + tokenizer
ARG HF_REPO="Maxi-Lein/Qwen3-Embedding-8B-onnx"
ENV HF_BASE_URL="https://huggingface.co/${HF_REPO}/resolve/main"

RUN mkdir -p models/Qwen3-Embedding-8B-onnx/onnx && \
    aria2c -c -x16 -s16 --max-connection-per-server=16 \
      --retry-wait=5 --max-tries=0 \
      --dir=models/Qwen3-Embedding-8B-onnx/onnx \
      --out=model.onnx \
      "${HF_BASE_URL}/onnx/model.onnx?download=1" && \
    aria2c -c -x16 -s16 --max-connection-per-server=16 \
      --retry-wait=5 --max-tries=0 \
      --dir=models/Qwen3-Embedding-8B-onnx/onnx \
      --out=model.onnx_data \
      "${HF_BASE_URL}/onnx/model.onnx_data?download=1" && \
    aria2c -c -x16 -s16 --max-connection-per-server=16 \
      --retry-wait=5 --max-tries=0 \
      --dir=models/Qwen3-Embedding-8B-onnx \
      --out=tokenizer.json \
      "${HF_BASE_URL}/tokenizer.json?download=1"

# ---------- runtime ----------
FROM gcr.io/distroless/cc-debian12:nonroot
WORKDIR /app

# Rust binary (crate name qwen_embed_8_rs)
COPY --from=builder /src/target/release/qwen_embed_8_rs /app/qwen_embed_8_rs

# ORT shared libs + OpenMP
COPY --from=ort /opt/onnxruntime/lib /opt/onnxruntime/lib
COPY --from=syslibs /usr/lib/x86_64-linux-gnu/libgomp.so.1 /usr/lib/x86_64-linux-gnu/libgomp.so.1

# Model + tokenizer in layout expected by main.rs
COPY --from=model /download/models /app/models

ENV ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so \
    LD_LIBRARY_PATH=/opt/onnxruntime/lib:/usr/lib/x86_64-linux-gnu

EXPOSE 8981
USER nonroot:nonroot
ENTRYPOINT ["/app/qwen_embed_8_rs"]
