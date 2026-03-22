//! BGE-M3 ONNX 임베딩 라이브러리
//!
//! BGE-M3 모델을 ONNX Runtime으로 인프로세스 실행하여
//! Dense / Sparse / ColBERT 3가지 임베딩을 생성합니다.

use anyhow::{Result, anyhow};
use ort::execution_providers::CPUExecutionProvider;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

/// BGE-M3 세 가지 임베딩 결과
pub struct BgeM3Output {
    pub dense: Vec<f32>,           // [1024] 정규화된 Dense 벡터
    pub sparse: HashMap<u32, f32>, // {token_id: weight} Sparse 가중치
    pub colbert: Vec<Vec<f32>>,    // [seq_len][1024] ColBERT 벡터들
}

/// ONNX Runtime 초기화 — 프로세스당 한 번 호출
pub fn init_ort() {
    ort::init().commit();
}

/// BGE-M3 ONNX 임베더
pub struct BgeM3Embedder {
    tokenizer: Tokenizer,
    session: Session,
}

impl BgeM3Embedder {
    pub fn new(model_path: impl AsRef<Path>, tokenizer_path: impl AsRef<Path>) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("토크나이저 로드 실패: {}", e))?;

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

        tracing::info!("BGE-M3 임베더 로딩 완료");
        Ok(Self { tokenizer, session })
    }

    /// 텍스트를 Dense/Sparse/ColBERT 벡터로 인코딩
    pub fn encode(&mut self, text: &str) -> Result<BgeM3Output> {
        let encoding = self.tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("토큰화 실패: {}", e))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
        let seq_len = input_ids.len();

        let input_ids_value = Value::from_array(([1, seq_len], input_ids.clone()))
            .map_err(|e| anyhow!("Input IDs error: {:?}", e))?;
        let attention_mask_value = Value::from_array(([1, seq_len], attention_mask))
            .map_err(|e| anyhow!("Attention mask error: {:?}", e))?;

        let outputs = self.session
            .run(vec![("input_ids", input_ids_value), ("attention_mask", attention_mask_value)])
            .map_err(|e| anyhow!("추론 실패: {:?}", e))?;

        // Dense: [batch=1, 1024]
        let (_shape, dense_data) = outputs["dense_vecs"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("dense_vecs 추출 실패: {:?}", e))?;
        let dense = dense_data[..1024].to_vec();

        // Sparse: [batch=1, token, 1] → HashMap<token_id, weight>
        let (_shape, sparse_data) = outputs["sparse_vecs"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("sparse_vecs 추출 실패: {:?}", e))?;
        let special_tokens: [i64; 4] = [0, 1, 2, 3];
        let mut sparse = HashMap::new();
        for (i, &token_id) in input_ids.iter().enumerate() {
            let weight = sparse_data[i];
            if weight > 0.0 && !special_tokens.contains(&token_id) {
                let tid = token_id as u32;
                let entry = sparse.entry(tid).or_insert(0.0f32);
                if weight > *entry { *entry = weight; }
            }
        }

        // ColBERT: [batch=1, token, 1024]
        let (colbert_shape, colbert_data) = outputs["colbert_vecs"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("colbert_vecs 추출 실패: {:?}", e))?;
        let colbert_tokens = colbert_shape[1] as usize;
        let mut colbert = Vec::with_capacity(colbert_tokens.saturating_sub(2));
        for i in 1..colbert_tokens.saturating_sub(1) {
            let start = i * 1024;
            colbert.push(colbert_data[start..start + 1024].to_vec());
        }

        Ok(BgeM3Output { dense, sparse, colbert })
    }

    /// Dense 벡터만 빠르게 반환 (RAG 검색용)
    pub fn encode_dense(&mut self, text: &str) -> Result<Vec<f32>> {
        Ok(self.encode(text)?.dense)
    }
}

// === 유사도 계산 함수들 ===

/// Dense: 코사인 유사도
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b + 1e-9)
}

/// Sparse: 공통 토큰의 가중치 곱 합산
pub fn sparse_dot_product(a: &HashMap<u32, f32>, b: &HashMap<u32, f32>) -> f32 {
    let (smaller, larger) = if a.len() < b.len() { (a, b) } else { (b, a) };
    smaller.iter()
        .filter_map(|(k, v)| larger.get(k).map(|w| v * w))
        .sum()
}

/// Multi-Vector (ColBERT): MaxSim
pub fn max_sim(query_vecs: &[Vec<f32>], doc_vecs: &[Vec<f32>]) -> f32 {
    if query_vecs.is_empty() || doc_vecs.is_empty() { return 0.0; }
    let sum: f32 = query_vecs.iter()
        .map(|q| doc_vecs.iter()
            .map(|d| cosine_similarity(q, d))
            .fold(f32::NEG_INFINITY, f32::max))
        .sum();
    sum / query_vecs.len() as f32
}
