use anyhow::{Result, anyhow};
use ort::execution_providers::CPUExecutionProvider;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

// === BGE-M3 세 가지 임베딩 결과 ===
struct BgeM3Output {
    dense: Vec<f32>,           // [1024] 정규화된 Dense 벡터
    sparse: HashMap<u32, f32>, // {token_id: weight} Sparse 가중치
    colbert: Vec<Vec<f32>>,    // [seq_len][1024] ColBERT 벡터들
}

struct BgeM3Embedder {
    tokenizer: Tokenizer,
    session: Session,
}

impl BgeM3Embedder {
    pub fn new(model_path: impl AsRef<Path>, tokenizer_path: impl AsRef<Path>) -> Result<Self> {
        println!(">> [1] 토크나이저 로딩...");
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("토크나이저 로드 실패: {}", e))?;

        println!(">> [2] ONNX 세션 생성...");
        let session = Session::builder()
            .map_err(|e| anyhow!("Builder error: {:?}", e))?
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| anyhow!("EP error: {:?}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("Opt error: {:?}", e))?
            .with_intra_threads(4)
            .map_err(|e| anyhow!("Thread error: {:?}", e))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("모델 로드 실패: {:?}", e))?;

        println!(">> [3] 로딩 완료!");
        Ok(Self { tokenizer, session })
    }

    pub fn encode(&mut self, text: &str) -> Result<BgeM3Output> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("토큰화 실패: {}", e))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let seq_len = input_ids.len();

        let input_ids_value = Value::from_array(([1, seq_len], input_ids.clone()))
            .map_err(|e| anyhow!("Input IDs error: {:?}", e))?;
        let attention_mask_value = Value::from_array(([1, seq_len], attention_mask))
            .map_err(|e| anyhow!("Attention mask error: {:?}", e))?;

        let outputs = self
            .session
            .run(vec![
                ("input_ids", input_ids_value),
                ("attention_mask", attention_mask_value),
            ])
            .map_err(|e| anyhow!("추론 실패: {:?}", e))?;

        // --- Dense: [batch=1, 1024] ---
        let (_shape, dense_data) = outputs["dense_vecs"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("dense_vecs 추출 실패: {:?}", e))?;
        let dense = dense_data[..1024].to_vec();

        // --- Sparse: [batch=1, token, 1] → HashMap<token_id, weight> ---
        let (_shape, sparse_data) = outputs["sparse_vecs"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("sparse_vecs 추출 실패: {:?}", e))?;

        let special_tokens: [i64; 4] = [0, 1, 2, 3]; // CLS, PAD, EOS, UNK
        let mut sparse = HashMap::new();
        for (i, &token_id) in input_ids.iter().enumerate() {
            let weight = sparse_data[i];
            if weight > 0.0 && !special_tokens.contains(&token_id) {
                let tid = token_id as u32;
                let entry = sparse.entry(tid).or_insert(0.0f32);
                if weight > *entry {
                    *entry = weight;
                }
            }
        }

        // --- ColBERT: [batch=1, token, 1024] ---
        let (colbert_shape, colbert_data) = outputs["colbert_vecs"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("colbert_vecs 추출 실패: {:?}", e))?;

        let colbert_tokens = colbert_shape[1] as usize;
        // [CLS](인덱스 0)와 [SEP](마지막 인덱스) 제외
        let mut colbert = Vec::with_capacity(colbert_tokens.saturating_sub(2));
        for i in 1..colbert_tokens.saturating_sub(1) {
            let start = i * 1024;
            colbert.push(colbert_data[start..start + 1024].to_vec());
        }

        Ok(BgeM3Output {
            dense,
            sparse,
            colbert,
        })
    }
}

// === 유사도 계산 함수들 ===

/// Dense: 코사인 유사도
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b + 1e-9)
}

/// Sparse: 공통 토큰의 가중치 곱 합산
fn sparse_dot_product(a: &HashMap<u32, f32>, b: &HashMap<u32, f32>) -> f32 {
    let (smaller, larger) = if a.len() < b.len() { (a, b) } else { (b, a) };
    smaller
        .iter()
        .filter_map(|(k, v)| larger.get(k).map(|w| v * w))
        .sum()
}

/// Multi-Vector (ColBERT): MaxSim
fn max_sim(query_vecs: &[Vec<f32>], doc_vecs: &[Vec<f32>]) -> f32 {
    if query_vecs.is_empty() || doc_vecs.is_empty() {
        return 0.0;
    }
    let sum: f32 = query_vecs
        .iter()
        .map(|q| {
            doc_vecs
                .iter()
                .map(|d| cosine_similarity(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum();
    sum / query_vecs.len() as f32
}

fn main() -> Result<()> {
    ort::init().commit();

    let model_path = "../models/bge-m3/model_quantized.onnx";
    let tokenizer_path = "../models/bge-m3/tokenizer.json";
    let mut embedder = BgeM3Embedder::new(model_path, tokenizer_path)?;

    let top_k = 3; // 1단계에서 가져올 후보 수

    // === 테스트 데이터 ===
    let query = "장삼풍의 태극권";
    let documents = [
        "장삼풍이 무당산에서 태극권을 창시했다",
        "옥교룡이 이모백에게 청명보검을 건넸다",
        "태극검법은 무당파의 진산지보이다",
        "화산파의 검법은 빠르고 날카롭다",
        "장삼풍은 소림사에서 무당파를 세웠다",
    ];

    println!("\n===== 2단계 Hybrid Retrieval + ColBERT Reranking =====");
    println!("쿼리: \"{}\"", query);
    println!(
        "전체 문서: {}개 → 1단계 후보: {}개 → 2단계 ColBERT 리랭킹\n",
        documents.len(),
        top_k
    );

    // 쿼리 인코딩
    let query_output = embedder.encode(query)?;

    // ===================================================
    // 1단계: Dense + Sparse로 후보 선별 (가볍고 빠름)
    // ===================================================
    println!("========== 1단계: Dense + Sparse 후보 선별 ==========\n");

    // Dense, Sparse 점수만 계산 (ColBERT는 아직 안 씀)
    let mut stage1_results: Vec<(usize, &str, f32, f32)> = Vec::new();
    for (idx, doc) in documents.iter().enumerate() {
        let doc_output = embedder.encode(doc)?;
        let dense_score = cosine_similarity(&query_output.dense, &doc_output.dense);
        let sparse_score = sparse_dot_product(&query_output.sparse, &doc_output.sparse);
        stage1_results.push((idx, doc, dense_score, sparse_score));
    }

    // Dense 상위 3개
    let mut dense_ranked = stage1_results.clone();
    dense_ranked.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    println!("--- Dense 상위 {}개 ---", top_k);
    for (i, (_, doc, score, _)) in dense_ranked.iter().take(top_k).enumerate() {
        println!("  {}위: {:.4} | {}", i + 1, score, doc);
    }

    // Sparse 상위 3개
    let mut sparse_ranked = stage1_results.clone();
    sparse_ranked.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    println!("--- Sparse 상위 {}개 ---", top_k);
    for (i, (_, doc, _, score)) in sparse_ranked.iter().take(top_k).enumerate() {
        println!("  {}위: {:.4} | {}", i + 1, score, doc);
    }

    // Dense + Sparse 합집합으로 후보 선별 (중복 제거)
    let mut candidate_indices: Vec<usize> = Vec::new();
    for &(idx, _, _, _) in dense_ranked.iter().take(top_k) {
        if !candidate_indices.contains(&idx) {
            candidate_indices.push(idx);
        }
    }
    for &(idx, _, _, _) in sparse_ranked.iter().take(top_k) {
        if !candidate_indices.contains(&idx) {
            candidate_indices.push(idx);
        }
    }

    println!("\n--- 합집합 후보: {}개 ---", candidate_indices.len());
    for &idx in &candidate_indices {
        println!("  [{}] {}", idx, documents[idx]);
    }

    // ===================================================
    // 2단계: ColBERT로 후보만 리랭킹 (정밀하지만 무거움)
    // ===================================================
    println!("\n========== 2단계: ColBERT 리랭킹 ==========\n");

    let mut reranked: Vec<(&str, f32)> = Vec::new();
    for &idx in &candidate_indices {
        let doc_output = embedder.encode(documents[idx])?;
        let colbert_score = max_sim(&query_output.colbert, &doc_output.colbert);
        reranked.push((documents[idx], colbert_score));
    }

    reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("--- 최종 순위 (ColBERT MaxSim) ---");
    for (i, (doc, score)) in reranked.iter().enumerate() {
        println!("  {}위: {:.4} | {}", i + 1, score, doc);
    }

    Ok(())
}
