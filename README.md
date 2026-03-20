# bge-m3-onnx-rust

BGE-M3 임베딩 모델의 **Dense, Sparse, ColBERT** 세 가지 Retrieval을 Rust에서 ONNX Runtime으로 구현한 샘플 프로젝트.

## 특징

- **Dense Retrieval**: 문장을 1024차원 벡터 1개로 압축, 코사인 유사도로 의미 검색
- **Sparse (Lexical) Retrieval**: 토큰별 가중치로 키워드 매칭 (학습된 BM25)
- **Multi-Vector (ColBERT) Retrieval**: 토큰 수준 정밀 매칭, MaxSim 계산
- **2단계 Hybrid Pipeline**: Dense + Sparse로 후보 선별 → ColBERT로 리랭킹
- INT8 양자화 모델 사용 (570MB)
- CPU 전용, GPU 불필요

## 실행 결과

```
===== 2단계 Hybrid Retrieval + ColBERT Reranking =====
쿼리: "장삼풍의 태극권"
전체 문서: 5개 → 1단계 후보: 3개 → 2단계 ColBERT 리랭킹

========== 1단계: Dense + Sparse 후보 선별 ==========
--- Dense 상위 3개 ---
  1위: 0.6967 | 장삼풍이 무당산에서 태극권을 창시했다
  2위: 0.5176 | 장삼풍은 소림사에서 무당파를 세웠다
  3위: 0.4571 | 태극검법은 무당파의 진산지보이다
--- Sparse 상위 3개 ---
  1위: 0.1875 | 장삼풍이 무당산에서 태극권을 창시했다
  2위: 0.1206 | 장삼풍은 소림사에서 무당파를 세웠다
  3위: 0.0544 | 태극검법은 무당파의 진산지보이다

========== 2단계: ColBERT 리랭킹 ==========
--- 최종 순위 (ColBERT MaxSim) ---
  1위: 0.7792 | 장삼풍이 무당산에서 태극권을 창시했다
  2위: 0.6030 | 장삼풍은 소림사에서 무당파를 세웠다
  3위: 0.4653 | 태극검법은 무당파의 진산지보이다
```

## 사전 준비

### 1. 모델 다운로드

[gpahal/bge-m3-onnx-int8](https://huggingface.co/gpahal/bge-m3-onnx-int8)에서 두 파일을 다운로드:

- `model_quantized.onnx` (570MB)
- `tokenizer.json` (17MB)

프로젝트 기준 `../models/bge-m3/` 경로에 배치하거나, `src/main.rs`에서 경로를 수정:

```
projects/
├── models/
│   └── bge-m3/
│       ├── model_quantized.onnx
│       └── tokenizer.json
└── bge-m3-onnx-rust/
    ├── src/main.rs
    └── ...
```

### 2. 빌드 환경 (Windows)

- Rust (stable, MSVC toolchain)
- Visual Studio Build Tools (MSVC 14.x)

### 빌드 & 실행

```powershell
cargo clean
cargo run --release
```

## 핵심 설정 (Windows MSVC)

`ort` 크레이트 prebuilt 바이너리와 CRT 링킹 방식을 일치시켜야 한다.

`.cargo/config.toml`:
```toml
[target.x86_64-pc-windows-msvc]
rustflags = ["-C", "target-feature=-crt-static"]

[env]
CFLAGS = "/MD"
CXXFLAGS = "/MD"
```

자세한 내용은 [ONNX_BUILD_GUIDE.md](ONNX_BUILD_GUIDE.md) 참조.

## ONNX 모델 선택 가이드

| 모델 | 출처 | 크기 | Dense | Sparse | ColBERT |
|-----|------|------|-------|--------|---------|
| `model_quantized.onnx` | gpahal/bge-m3-onnx-int8 | 570MB | ✅ | ✅ | ✅ |
| `sentence_transformers_int8.onnx` | Xenova/bge-m3 | 569MB | ✅ | ❌ | ❌ |
| `bge_m3_model.onnx` | yuniko-software/bge-m3-onnx | 2.27GB | ✅ | ✅ | ✅ |

**추천**: `gpahal/bge-m3-onnx-int8` — INT8 + 3가지 출력 모두 포함

## 문서

- [ONNX_BUILD_GUIDE.md](ONNX_BUILD_GUIDE.md) — Windows 빌드 설정, CRT 문제 해결, 트러블슈팅
- [BGE_M3_RETRIEVAL.md](BGE_M3_RETRIEVAL.md) — Dense, Sparse, ColBERT 방식 상세 설명

## 의존성

| 크레이트 | 버전 | 용도 |
|---------|------|------|
| `ort` | 2.0.0-rc.12 | ONNX Runtime Rust 바인딩 |
| `tokenizers` | 0.21.0 | HuggingFace 토크나이저 |
| `ndarray` | 0.16.1 | 다차원 배열 |
| `anyhow` | 1.0 | 에러 처리 |

## 참고

- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) — 원본 모델
- [BGE-M3 논문](https://arxiv.org/abs/2402.03216) — Multi-Lingual, Multi-Functionality, Multi-Granularity
- [ort 크레이트](https://ort.pyke.io/) — Rust ONNX Runtime 바인딩

## License

MIT
