# BGE-M3 ONNX Runtime - Rust 빌드 가이드 (Windows)

## 환경 정보

- OS: Windows (x86_64-pc-windows-msvc)
- Rust 크레이트: `ort` 2.0.0-rc.12
- Visual Studio: 18 Community, MSVC 14.50.35717
- GPU: RTX 2070 (본 프로젝트는 CPU 전용)

## 핵심 문제와 해결

### 1. onnxruntime.dll 버전 충돌 (Hang 현상)

`ort` 크레이트의 `load-dynamic` feature를 사용하면 런타임에 DLL을 탐색한다.
이때 `C:\Windows\system32\onnxruntime.dll` (Windows 내장, v1.17, 10.1MB)이 먼저 로딩되는데,
`ort` 2.0.0-rc.12는 v1.20.x를 기대하므로 버전 불일치로 세션 생성 시 hang이 발생한다.

**해결**: `load-dynamic` feature를 제거하고 정적 링크를 사용한다.

```toml
# Cargo.toml - load-dynamic 제거
ort = { version = "2.0.0-rc.12", features = ["ndarray"] }
```

정적 링크 시 onnxruntime이 EXE에 포함되므로 시스템 DLL과 충돌하지 않는다.

### 2. CRT 링킹 불일치 (LNK2038 오류)

`load-dynamic` 제거 후 빌드하면 다음 링커 오류가 발생한다:

```
error LNK2038: 'RuntimeLibrary'에 대해 불일치가 검색되었습니다.
'MD_DynamicRelease' 값이 'MT_StaticRelease' 값과 일치하지 않습니다.
```

**원인**: CRT(C Runtime Library) 링킹 방식의 충돌

| 구성요소 | CRT 방식 | 의미 |
|---------|---------|------|
| ort prebuilt `onnxruntime.lib` | `/MD` (동적 CRT) | `msvcrt.dll` 참조 |
| Rust MSVC 타겟 기본값 | `/MT` (정적 CRT) | CRT 코드를 EXE에 포함 |
| esaxx_rs (C++ 의존성) | `/MT` (정적 CRT) | Rust 기본값 따름 |

한 프로그램 안에서 `/MD`와 `/MT`를 혼용하면 메모리 관리자가 충돌한다.
ort prebuilt가 `/MD`로 빌드되었으므로 나머지도 `/MD`로 통일해야 한다.

**해결 1**: `.cargo/config.toml`에서 Rust의 CRT를 동적으로 변경

```toml
# .cargo/config.toml
[target.x86_64-pc-windows-msvc]
rustflags = ["-C", "target-feature=-crt-static"]

[env]
CFLAGS = "/MD"
CXXFLAGS = "/MD"
```

**각 설정의 역할**:

- `target-feature=-crt-static`: Rust 컴파일러에게 동적 CRT(`/MD`)를 사용하라고 지시.
  Rust MSVC 타겟의 기본값이 `+crt-static`(MT)이므로 **명시적으로 비활성화** 필요.
  단순히 줄을 삭제하면 기본값(MT)이 적용되어 효과 없음.

- `CFLAGS="/MD"`: `cc` 크레이트가 C 파일(`.c`)을 컴파일할 때 MSVC에 `/MD` 전달.
  `onig_sys` 등의 C 의존성에 적용됨.

- `CXXFLAGS="/MD"`: `cc` 크레이트가 C++ 파일(`.cpp`)을 컴파일할 때 MSVC에 `/MD` 전달.
  `esaxx_rs` 등의 C++ 의존성에 적용됨.

**해결 2**: 환경변수로 직접 설정 (config.toml 대신)

```powershell
$env:CFLAGS="/MD"
$env:CXXFLAGS="/MD"
cargo run --release
```

## CRT(C Runtime Library) 개념 정리

CRT는 C/C++ 프로그램의 기본 함수 모음 (`malloc`, `printf`, `fopen` 등).
Windows에서 두 가지 링크 방식이 있다:

| 방식 | 플래그 | 동작 | 장단점 |
|-----|-------|------|-------|
| 동적 CRT | `/MD` | `msvcrt.dll` 참조 | 바이너리 작음, Windows 기본 내장 |
| 정적 CRT | `/MT` | CRT 코드를 EXE에 포함 | 독립 실행 가능, 바이너리 큼 |

**혼용 불가**: 한 프로그램 안에서 `/MD`와 `/MT`를 섞으면,
두 개의 서로 다른 메모리 관리자가 동작하여 `malloc`/`free` 충돌 발생.

## 최종 빌드 설정

### Cargo.toml

```toml
[package]
name = "bge-m3-onnx-sample"
version = "0.1.0"
edition = "2024"

[dependencies]
ort = { version = "2.0.0-rc.12", features = ["ndarray"] }
tokenizers = "0.21.0"
ndarray = "0.16.1"
anyhow = "1.0.95"
serde = { version = "1.0.217", features = ["derive"] }
serde_json = "1.0.138"
tokio = { version = "1.43.0", features = ["full"] }
```

### .cargo/config.toml

```toml
[target.x86_64-pc-windows-msvc]
rustflags = ["-C", "target-feature=-crt-static"]

[env]
CFLAGS = "/MD"
CXXFLAGS = "/MD"
```

### 빌드 명령어

```powershell
cargo clean        # 이전 빌드 캐시 제거 (CRT 변경 시 필수)
cargo run --release
```

## ONNX 모델 선택

### 모델 비교

| 모델 | 출처 | 크기 | Dense | Sparse | ColBERT | 양자화 |
|-----|------|------|-------|--------|---------|-------|
| `model_int8.onnx` | Xenova/bge-m3 | 568MB | ✅ (`last_hidden_state`에서 수동 추출) | ❌ | ❌ | INT8 |
| `sentence_transformers_int8.onnx` | Xenova/bge-m3 | 569MB | ✅ (`sentence_embedding`) | ❌ | ❌ | INT8 |
| `model_quantized.onnx` | gpahal/bge-m3-onnx-int8 | 570MB | ✅ (`dense_vecs`) | ✅ (`sparse_vecs`) | ✅ (`colbert_vecs`) | INT8 |
| `bge_m3_model.onnx` | yuniko-software/bge-m3-onnx | 2.27GB | ✅ | ✅ | ✅ | FP32 |

**추천**: `gpahal/bge-m3-onnx-int8`의 `model_quantized.onnx`
- INT8 양자화로 메모리 효율적
- Dense, Sparse, ColBERT 세 가지 출력 모두 포함
- 별도 가중치 추출 불필요

### model_quantized.onnx 출력 구조

```
입력:
  input_ids:      [batch_size, sequence_length]  (dtype: int64)
  attention_mask: [batch_size, sequence_length]  (dtype: int64)

출력:
  dense_vecs:   [batch_size, 1024]              → Dense Retrieval용 (정규화 완료)
  sparse_vecs:  [batch_size, token, 1]          → Sparse Retrieval용 (ReLU 적용 완료)
  colbert_vecs: [batch_size, token, 1024]       → ColBERT Retrieval용 (정규화 완료)
```

## GraphOptimizationLevel 설정

```rust
.with_optimization_level(GraphOptimizationLevel::Level3)
```

Level3는 ONNX Runtime이 연산 그래프를 재구성하여 추론 속도를 높인다.
(operator fusion, 연산 순서 변경 등)

부동소수점 연산 순서가 바뀌므로 `Disable`과 소수점 6~7자리에서 미세한 차이가 발생하지만,
임베딩 검색(코사인 유사도)에는 영향 없음. **Level3 사용 권장**.

## CPUExecutionProvider 명시

```rust
.with_execution_providers([CPUExecutionProvider::default().build()])
```

EP를 명시하지 않으면 ort prebuilt 바이너리에 포함된 DirectML EP가
초기화를 시도할 수 있다. GPU 드라이버 문제 시 hang 발생 가능.
CPU 전용이면 반드시 명시할 것.

## 트러블슈팅 체크리스트

| 증상 | 원인 | 해결 |
|-----|------|------|
| ONNX 세션 생성 시 hang (3분+) | 시스템 `onnxruntime.dll` v1.17 로딩 | `load-dynamic` 제거 → 정적 링크 |
| ONNX 세션 생성 시 hang | DirectML EP 초기화 실패 | `CPUExecutionProvider` 명시 |
| LNK2038 RuntimeLibrary 불일치 | `/MD` vs `/MT` CRT 충돌 | `.cargo/config.toml`에 `-crt-static` + `CFLAGS`/`CXXFLAGS` |
| `cargo clean` 후에도 MT 유지 | Rust MSVC 기본값이 `+crt-static` | `-crt-static` **명시적 비활성화** 필요 |
| ColBERT 인덱스 범위 초과 | seq_len과 colbert 토큰 수 불일치 | `colbert_shape[1]`에서 실제 크기 사용 |

## 프로젝트 파일 구조

```
projects/
├── models/
│   └── bge-m3/
│       ├── model_quantized.onnx  # gpahal/bge-m3-onnx-int8 (570MB, INT8)
│       └── tokenizer.json        # XLM-RoBERTa 토크나이저
└── bge-m3-onnx-rust/
    ├── .cargo/
    │   └── config.toml           # CRT 동적 링크 + CFLAGS/CXXFLAGS 설정
    ├── src/
    │   └── main.rs               # Dense + Sparse + ColBERT 비교 샘플
    ├── Cargo.toml
    ├── ONNX_BUILD_GUIDE.md       # 이 문서
    └── BGE_M3_RETRIEVAL.md       # Retrieval 방식 상세 설명
```
