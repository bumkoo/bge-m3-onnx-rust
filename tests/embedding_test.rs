//! BGE-M3 ONNX 통합 테스트
//!
//! 모델 파일 필요: ../models/bge-m3/model_quantized.onnx, tokenizer.json
//! `cargo test` 로 실행

use bge_m3_onnx_rust::{
    BgeM3Embedder, cosine_similarity, sparse_dot_product, max_sim,
};
use std::sync::Once;

static INIT: Once = Once::new();

fn init_ort() {
    INIT.call_once(|| { ort::init().commit(); });
}

fn create_embedder() -> BgeM3Embedder {
    init_ort();
    BgeM3Embedder::new(
        "../models/bge-m3/model_quantized.onnx",
        "../models/bge-m3/tokenizer.json",
    ).expect("임베더 생성 실패 — 모델 파일 확인")
}

// ===== Dense 임베딩 기본 테스트 =====

#[test]
fn test_dense_embedding_dimension() {
    let mut embedder = create_embedder();
    let output = embedder.encode("테스트 문장").unwrap();
    assert_eq!(output.dense.len(), 1024, "Dense 벡터는 1024차원이어야 함");
}

#[test]
fn test_dense_embedding_normalized() {
    let mut embedder = create_embedder();
    let output = embedder.encode("무당검법은 무당파의 검술이다").unwrap();
    let norm: f32 = output.dense.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01, "Dense 벡터는 정규화되어야 함: norm={}", norm);
}

#[test]
fn test_encode_dense_shortcut() {
    let mut embedder = create_embedder();
    let dense = embedder.encode_dense("테스트").unwrap();
    assert_eq!(dense.len(), 1024);
}

// ===== Sparse 임베딩 테스트 =====

#[test]
fn test_sparse_embedding_nonempty() {
    let mut embedder = create_embedder();
    let output = embedder.encode("장삼풍이 태극권을 창시했다").unwrap();
    assert!(!output.sparse.is_empty(), "Sparse 벡터는 비어있으면 안 됨");
    // 모든 가중치는 양수
    for (&_tid, &weight) in &output.sparse {
        assert!(weight > 0.0, "Sparse 가중치는 양수여야 함");
    }
}

// ===== ColBERT 임베딩 테스트 =====

#[test]
fn test_colbert_embedding_shape() {
    let mut embedder = create_embedder();
    let output = embedder.encode("무당산 절벽").unwrap();
    assert!(!output.colbert.is_empty(), "ColBERT 벡터는 비어있으면 안 됨");
    for vec in &output.colbert {
        assert_eq!(vec.len(), 1024, "ColBERT 각 토큰 벡터는 1024차원");
    }
}

// ===== 유사도 함수 테스트 =====

#[test]
fn test_cosine_similarity_identical() {
    let v = vec![1.0, 0.0, 0.0];
    assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    assert!(cosine_similarity(&a, &b).abs() < 1e-6);
}

#[test]
fn test_sparse_dot_product_disjoint() {
    let a = [(1u32, 0.5f32)].into_iter().collect();
    let b = [(2u32, 0.8f32)].into_iter().collect();
    assert_eq!(sparse_dot_product(&a, &b), 0.0);
}

#[test]
fn test_sparse_dot_product_overlap() {
    let a = [(1u32, 0.5f32), (2, 0.3)].into_iter().collect();
    let b = [(1u32, 0.8f32), (3, 0.2)].into_iter().collect();
    let expected = 0.5 * 0.8;  // 토큰 1만 공통
    assert!((sparse_dot_product(&a, &b) - expected).abs() < 1e-6);
}

// ===== 검색 품질 테스트 (Dense) =====

#[test]
fn test_dense_retrieval_ranking() {
    let mut embedder = create_embedder();

    let query = "장삼풍의 태극권";
    let documents = [
        "장삼풍이 무당산에서 태극권을 창시했다",    // 가장 관련
        "옥교룡이 이모백에게 청명보검을 건넸다",    // 무관
        "태극검법은 무당파의 진산지보이다",          // 부분 관련
        "화산파의 검법은 빠르고 날카롭다",          // 무관
        "장삼풍은 소림사에서 무당파를 세웠다",      // 부분 관련
    ];

    let query_dense = embedder.encode_dense(query).unwrap();
    let mut scores: Vec<(usize, f32)> = documents.iter()
        .enumerate()
        .map(|(i, doc)| {
            let doc_dense = embedder.encode_dense(doc).unwrap();
            (i, cosine_similarity(&query_dense, &doc_dense))
        })
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // 1위는 "장삼풍이 무당산에서 태극권을 창시했다" (인덱스 0)
    assert_eq!(scores[0].0, 0, "1위가 가장 관련 문서여야 함");
    // 1위 점수 > 0.5
    assert!(scores[0].1 > 0.5, "1위 유사도가 0.5 이상이어야 함: {}", scores[0].1);
}

// ===== ColBERT 리랭킹 테스트 =====

#[test]
fn test_colbert_reranking_improves() {
    let mut embedder = create_embedder();

    let query_output = embedder.encode("장삼풍의 태극권").unwrap();
    let doc_a = embedder.encode("장삼풍이 무당산에서 태극권을 창시했다").unwrap();
    let doc_b = embedder.encode("화산파의 검법은 빠르고 날카롭다").unwrap();

    let score_a = max_sim(&query_output.colbert, &doc_a.colbert);
    let score_b = max_sim(&query_output.colbert, &doc_b.colbert);

    assert!(score_a > score_b,
        "ColBERT: 관련 문서({:.4}) > 무관 문서({:.4})", score_a, score_b);
}

// ===== 한글/영문 혼합 테스트 =====

#[test]
fn test_multilingual_embedding() {
    let mut embedder = create_embedder();
    let ko = embedder.encode_dense("무당파 검법").unwrap();
    let en = embedder.encode_dense("Wudang sword technique").unwrap();
    let unrelated = embedder.encode_dense("오늘 날씨가 좋다").unwrap();

    let sim_ko_en = cosine_similarity(&ko, &en);
    let sim_ko_unrelated = cosine_similarity(&ko, &unrelated);

    assert!(sim_ko_en > sim_ko_unrelated,
        "한영 유사도({:.4}) > 무관 유사도({:.4})", sim_ko_en, sim_ko_unrelated);
}
