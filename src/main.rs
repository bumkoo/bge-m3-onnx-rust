//! BGE-M3 ONNX 임베딩 테스트 바이너리
//! 2단계 Hybrid Retrieval + ColBERT Reranking 데모

use anyhow::Result;
use bge_m3_onnx_rust::{
    BgeM3Embedder, cosine_similarity, sparse_dot_product, max_sim,
};

fn main() -> Result<()> {
    bge_m3_onnx_rust::init_ort();

    let model_path = "../models/bge-m3/model_quantized.onnx";
    let tokenizer_path = "../models/bge-m3/tokenizer.json";
    let mut embedder = BgeM3Embedder::new(model_path, tokenizer_path)?;

    let top_k = 3;

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
    println!("전체 문서: {}개 → 1단계 후보: {}개 → 2단계 ColBERT 리랭킹\n",
        documents.len(), top_k);

    let query_output = embedder.encode(query)?;

    // === 1단계: Dense + Sparse로 후보 선별 ===
    println!("========== 1단계: Dense + Sparse 후보 선별 ==========\n");

    let mut stage1_results: Vec<(usize, &str, f32, f32)> = Vec::new();
    for (idx, doc) in documents.iter().enumerate() {
        let doc_output = embedder.encode(doc)?;
        let dense_score = cosine_similarity(&query_output.dense, &doc_output.dense);
        let sparse_score = sparse_dot_product(&query_output.sparse, &doc_output.sparse);
        stage1_results.push((idx, doc, dense_score, sparse_score));
    }

    let mut dense_ranked = stage1_results.clone();
    dense_ranked.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    println!("--- Dense 상위 {}개 ---", top_k);
    for (i, (_, doc, score, _)) in dense_ranked.iter().take(top_k).enumerate() {
        println!("  {}위: {:.4} | {}", i + 1, score, doc);
    }

    let mut sparse_ranked = stage1_results.clone();
    sparse_ranked.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    println!("--- Sparse 상위 {}개 ---", top_k);
    for (i, (_, doc, _, score)) in sparse_ranked.iter().take(top_k).enumerate() {
        println!("  {}위: {:.4} | {}", i + 1, score, doc);
    }

    let mut candidate_indices: Vec<usize> = Vec::new();
    for &(idx, _, _, _) in dense_ranked.iter().take(top_k) {
        if !candidate_indices.contains(&idx) { candidate_indices.push(idx); }
    }
    for &(idx, _, _, _) in sparse_ranked.iter().take(top_k) {
        if !candidate_indices.contains(&idx) { candidate_indices.push(idx); }
    }

    println!("\n--- 합집합 후보: {}개 ---", candidate_indices.len());
    for &idx in &candidate_indices {
        println!("  [{}] {}", idx, documents[idx]);
    }

    // === 2단계: ColBERT로 후보만 리랭킹 ===
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
