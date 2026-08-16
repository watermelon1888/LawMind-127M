"""复现历史单 query、top-4 retrieval 参数比较并生成 Markdown 报告。"""

import argparse
import json
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rag.knowledge import ArticleRepository
from rag.retrieval.dense import DenseSearcher
from rag.retrieval.loader import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    _resolve_device,
    validate_loaded_artifacts,
)
from rag.retrieval.rerank import WindowReranker
from rag.retrieval.sparse import SparseSearcher


RAG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_SET = RAG_DIR / "eval" / "eval_set.jsonl"
DEFAULT_ARTICLE_INDEX = RAG_DIR / "chunk" / "article_index.jsonl"
DEFAULT_ARTIFACT_DIR = RAG_DIR / "retrieval" / "artifacts"
DEFAULT_REPORT = RAG_DIR / "eval" / "results" / "retrieval-parameter-selection.md"
EXPECTED_CASE_COUNT = 140
OBSERVATION_DEPTHS = (4, 5, 10)


@dataclass(frozen=True)
class RetrievalCase:
    """一条只使用原始问题和 canonical GT 的检索评估样本。"""

    case_id: str
    query: str
    gt_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalMetrics:
    """一组排名在固定观察深度上的宏平均指标。"""

    question_count: int
    any_hit_count: int
    complete_hit_count: int
    gt_coverage: float
    mrr: float

    @property
    def any_hit(self):
        return self.any_hit_count / self.question_count

    @property
    def complete_hit(self):
        return self.complete_hit_count / self.question_count


@dataclass(frozen=True)
class ConfigurationResult:
    """一个候选排名配置及其多观察深度结果。"""

    name: str
    rankings: dict[str, tuple[str, ...]]
    metrics: dict[int, RetrievalMetrics]


@dataclass(frozen=True)
class RerankCombinationResult:
    """一组粗排参数与候选池经过 reranker 后的诊断结果。"""

    dense_top_k: int
    sparse_top_k: int
    rrf_k: int
    candidate_pool: int
    pool_metrics: RetrievalMetrics
    reranked: ConfigurationResult
    missing_from_pool: tuple[str, ...]
    demoted_after_rerank: tuple[str, ...]

    @property
    def name(self):
        return (
            f"dense={self.dense_top_k}, sparse={self.sparse_top_k}, "
            f"rrf_k={self.rrf_k}, pool={self.candidate_pool}"
        )


def load_retrieval_cases(eval_set_path, repository):
    """加载 legal_query + answer 样本，并把 GT 解析为 chunk_id。"""
    cases = []
    with Path(eval_set_path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"开发集第 {line_number} 行不是有效 JSON") from exc
            if item.get("query_type") != "legal_query" or item.get(
                "expected_action"
            ) != "answer":
                continue
            query = item.get("query_original")
            gt_articles = item.get("gt_articles")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"{item.get('id')} 缺少 query_original")
            if not isinstance(gt_articles, list) or not gt_articles:
                raise ValueError(f"{item.get('id')} 缺少 gt_articles")
            gt_chunk_ids = []
            for reference in gt_articles:
                if not isinstance(reference, dict):
                    raise ValueError(f"{item.get('id')} 的 GT 格式无效")
                article = repository.lookup(
                    reference.get("law_name"),
                    reference.get("article_no"),
                )
                if article is None:
                    raise ValueError(f"{item.get('id')} 的 GT 无法解析: {reference!r}")
                gt_chunk_ids.append(article.chunk_id)
            if len(set(gt_chunk_ids)) != len(gt_chunk_ids):
                raise ValueError(f"{item.get('id')} 包含重复 GT")
            cases.append(
                RetrievalCase(
                    case_id=item["id"],
                    query=query,
                    gt_chunk_ids=tuple(gt_chunk_ids),
                )
            )
    if len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError(
            f"目标切片应为 {EXPECTED_CASE_COUNT} 条，实际为 {len(cases)} 条"
        )
    return tuple(cases)


def evaluate_rankings(cases, rankings, *, k):
    """计算 any-hit、GT coverage、complete-hit 和首个 GT 的 MRR。"""
    case_values = tuple(cases)
    if not case_values:
        raise ValueError("cases 不能为空")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k 必须是正整数")

    any_hits = 0
    complete_hits = 0
    coverage_sum = 0.0
    reciprocal_rank_sum = 0.0
    for case in case_values:
        ranking = tuple(rankings[case.case_id])[:k]
        if len(set(ranking)) != len(ranking):
            raise ValueError(f"{case.case_id} 的排名包含重复 chunk_id")
        gt = set(case.gt_chunk_ids)
        matched = gt.intersection(ranking)
        if matched:
            any_hits += 1
            first_rank = min(
                index
                for index, chunk_id in enumerate(ranking, start=1)
                if chunk_id in gt
            )
            reciprocal_rank_sum += 1.0 / first_rank
        if matched == gt:
            complete_hits += 1
        coverage_sum += len(matched) / len(gt)

    question_count = len(case_values)
    return RetrievalMetrics(
        question_count=question_count,
        any_hit_count=any_hits,
        complete_hit_count=complete_hits,
        gt_coverage=coverage_sum / question_count,
        mrr=reciprocal_rank_sum / question_count,
    )


def weighted_rrf(
    dense_results,
    sparse_results,
    *,
    dense_top_k,
    sparse_top_k,
    rrf_k,
    dense_weight,
    sparse_weight,
):
    """仅供评估器比较固定路线权重，不改变运行时融合实现。"""
    for name, value in (
        ("dense_top_k", dense_top_k),
        ("sparse_top_k", sparse_top_k),
        ("rrf_k", rrf_k),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} 必须是正整数")
    for name, value in (
        ("dense_weight", dense_weight),
        ("sparse_weight", sparse_weight),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} 必须是非负数")
    if dense_weight == 0 and sparse_weight == 0:
        raise ValueError("两路权重不能同时为零")

    scores = {}
    for results, top_k, weight in (
        (dense_results, dense_top_k, dense_weight),
        (sparse_results, sparse_top_k, sparse_weight),
    ):
        if weight == 0:
            continue
        seen = set()
        for rank, item in enumerate(tuple(results)[:top_k], start=1):
            if item.chunk_id in seen:
                raise ValueError("单路结果包含重复 chunk_id")
            seen.add(item.chunk_id)
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + weight / (
                rrf_k + rank
            )
    return tuple(
        chunk_id
        for chunk_id, _ in sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )


def build_fusion_rankings(
    cases,
    dense_results,
    sparse_results,
    *,
    dense_top_k=30,
    sparse_top_k=30,
    rrf_k=60,
    dense_weight=1,
    sparse_weight=1,
):
    """为全部评估题构造一个固定参数的 RRF 排名。"""
    return {
        case.case_id: weighted_rrf(
            dense_results[case.case_id],
            sparse_results[case.case_id],
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        for case in cases
    }


def build_configuration_result(name, cases, rankings):
    """计算一个配置在全部观察深度上的结果。"""
    return ConfigurationResult(
        name=name,
        rankings=rankings,
        metrics={
            k: evaluate_rankings(cases, rankings, k=k)
            for k in OBSERVATION_DEPTHS
        },
    )


def _collect_rankings(searcher, cases, *, top_k, label):
    started = time.perf_counter()
    results = {}
    for index, case in enumerate(cases, start=1):
        results[case.case_id] = tuple(searcher.search(case.query, top_k=top_k))
        if index % 20 == 0 or index == len(cases):
            print(f"{label}: {index}/{len(cases)}")
    return results, time.perf_counter() - started


def _load_components(*, repository, artifact_dir, device, batch_size):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import faiss
    from sentence_transformers import CrossEncoder, SentenceTransformer

    directory = Path(artifact_dir)
    dense_metadata = json.loads(
        (directory / "law_dense_meta.json").read_text(encoding="utf-8")
    )
    if dense_metadata.get("embedding_model") != DEFAULT_EMBEDDING_MODEL:
        raise ValueError("dense 产物记录的 embedding 模型与评估模型不一致")
    dense_index = faiss.read_index(str(directory / "law_dense.faiss"))
    with (directory / "law_sparse.pkl").open("rb") as source:
        sparse_payload = pickle.load(source)

    actual_device = _resolve_device(device)
    encoder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL, device=actual_device)
    cross_encoder = CrossEncoder(
        DEFAULT_RERANKER_MODEL,
        max_length=512,
        device=actual_device,
        tokenizer_args={"local_files_only": True},
        automodel_args={"local_files_only": True},
    )
    sparse_metadata = sparse_payload["metadata"]
    bm25 = sparse_payload["bm25"]
    validate_loaded_artifacts(
        repository=repository,
        dense_index=dense_index,
        dense_metadata=dense_metadata,
        sparse_metadata=sparse_metadata,
        embedding_dimension=encoder.get_sentence_embedding_dimension(),
        bm25_corpus_size=getattr(bm25, "corpus_size", None),
    )
    return (
        DenseSearcher(
            index=dense_index,
            position_to_window=dense_metadata["position_to_window"],
            encoder=encoder,
        ),
        SparseSearcher(
            bm25=bm25,
            position_to_chunk_id=sparse_metadata["position_to_chunk_id"],
        ),
        WindowReranker(
            model=cross_encoder,
            tokenizer=cross_encoder.tokenizer,
            max_length=512,
            overlap_tokens=64,
            batch_size=batch_size,
        ),
        actual_device,
        dense_metadata,
    )


def _rerank_requirements(cases, requirements, repository, reranker):
    """对多组候选取并集打分，再按各组原始排名分别精排。"""
    if not requirements:
        raise ValueError("requirements 不能为空")
    started = time.perf_counter()
    reranked = {name: {} for name in requirements}
    unique_candidate_count = 0
    for index, case in enumerate(cases, start=1):
        candidates_by_name = {
            name: tuple(rankings[case.case_id])[:pool]
            for name, (rankings, pool) in requirements.items()
        }
        union = tuple(
            dict.fromkeys(
                chunk_id
                for candidates in candidates_by_name.values()
                for chunk_id in candidates
            )
        )
        articles = tuple(repository.get_by_chunk_id(chunk_id) for chunk_id in union)
        scores = reranker.score(case.query, articles)
        score_by_chunk = dict(zip(union, scores))
        unique_candidate_count += len(union)
        for name, candidates in candidates_by_name.items():
            original_rank = {
                chunk_id: rank for rank, chunk_id in enumerate(candidates)
            }
            reranked[name][case.case_id] = tuple(
                sorted(
                    candidates,
                    key=lambda chunk_id: (
                        -score_by_chunk[chunk_id],
                        original_rank[chunk_id],
                        chunk_id,
                    ),
                )
            )
        if index % 10 == 0 or index == len(cases):
            print(f"reranker: {index}/{len(cases)}")
    return reranked, time.perf_counter() - started, unique_candidate_count


def classify_rerank_failures(cases, coarse_rankings, reranked_rankings, *, pool):
    """区分 GT 未完整进入候选池和进入后被精排排出 top-4。"""
    missing_from_pool = []
    demoted_after_rerank = []
    for case in cases:
        gt = set(case.gt_chunk_ids)
        if not gt.issubset(tuple(coarse_rankings[case.case_id])[:pool]):
            missing_from_pool.append(case.case_id)
        elif not gt.issubset(tuple(reranked_rankings[case.case_id])[:4]):
            demoted_after_rerank.append(case.case_id)
    return tuple(missing_from_pool), tuple(demoted_after_rerank)


def _paired_complete_delta(cases, result, baseline, *, k=4):
    gained = 0
    lost = 0
    for case in cases:
        gt = set(case.gt_chunk_ids)
        candidate_complete = gt.issubset(result.rankings[case.case_id][:k])
        baseline_complete = gt.issubset(baseline.rankings[case.case_id][:k])
        if candidate_complete and not baseline_complete:
            gained += 1
        elif baseline_complete and not candidate_complete:
            lost += 1
    return gained, lost


def _percent(value):
    return f"{value * 100:.2f}%"


def _result_table(cases, results, baseline):
    lines = [
        "| 配置 | complete@4 | 相对当前新增/丢失 | coverage@4 | any@4 | MRR@4 | complete@5 | complete@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        at4 = result.metrics[4]
        gained, lost = _paired_complete_delta(cases, result, baseline)
        lines.append(
            f"| {result.name} | {_percent(at4.complete_hit)} "
            f"({at4.complete_hit_count}/{at4.question_count}) | +{gained}/-{lost} | "
            f"{_percent(at4.gt_coverage)} | {_percent(at4.any_hit)} | "
            f"{at4.mrr:.4f} | {_percent(result.metrics[5].complete_hit)} | "
            f"{_percent(result.metrics[10].complete_hit)} |"
        )
    return "\n".join(lines)


def _group_table(cases, results):
    groups = (
        ("单 GT", tuple(case for case in cases if len(case.gt_chunk_ids) == 1)),
        ("多 GT", tuple(case for case in cases if len(case.gt_chunk_ids) > 1)),
    )
    lines = [
        "| 配置 | 分组 | 题数 | complete@4 | coverage@4 | any@4 | MRR@4 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for group_name, group_cases in groups:
            metrics = evaluate_rankings(group_cases, result.rankings, k=4)
            lines.append(
                f"| {result.name} | {group_name} | {metrics.question_count} | "
                f"{_percent(metrics.complete_hit)} | {_percent(metrics.gt_coverage)} | "
                f"{_percent(metrics.any_hit)} | {metrics.mrr:.4f} |"
            )
    return "\n".join(lines)


def _rerank_combination_table(results):
    lines = [
        "| 配置 | 候选池 complete | 候选池 Recall | 精排 complete@4 | 精排 Recall@4 | 精排 any@4 | 精排 MRR@4 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        at4 = result.reranked.metrics[4]
        lines.append(
            f"| {result.name} | {_percent(result.pool_metrics.complete_hit)} "
            f"({result.pool_metrics.complete_hit_count}/{result.pool_metrics.question_count}) | "
            f"{_percent(result.pool_metrics.gt_coverage)} | "
            f"{_percent(at4.complete_hit)} "
            f"({at4.complete_hit_count}/{at4.question_count}) | "
            f"{_percent(at4.gt_coverage)} | {_percent(at4.any_hit)} | {at4.mrr:.4f} |"
        )
    return "\n".join(lines)


def _rerank_failure_details(results):
    sections = []
    for result in results:
        missing_ids = "、".join(result.missing_from_pool) or "无"
        demoted_ids = "、".join(result.demoted_after_rerank) or "无"
        sections.append(
            f"### {result.name}\n\n"
            f"- GT 未完整进入 candidate pool（{len(result.missing_from_pool)} 题）："
            f"{missing_ids}\n"
            f"- GT 已完整进入 pool、但精排后未完整保留在 top-4（{len(result.demoted_after_rerank)} 题）："
            f"{demoted_ids}"
        )
    return "\n\n".join(sections)


def _best_names(results):
    best_key = max(
        (
            result.metrics[4].complete_hit_count,
            result.metrics[4].gt_coverage,
            result.metrics[4].mrr,
        )
        for result in results
    )
    return [
        result.name
        for result in results
        if (
            result.metrics[4].complete_hit_count,
            result.metrics[4].gt_coverage,
            result.metrics[4].mrr,
        )
        == best_key
    ]


def render_report(
    *,
    cases,
    dense_result,
    sparse_result,
    current_fusion,
    dense_top_results,
    sparse_top_results,
    rrf_k_results,
    weight_results,
    rerank_results,
    rerank_combinations,
    rerank_unique_candidate_count,
    timings,
    device,
    batch_size,
    dense_metadata,
):
    """把本次参数比较渲染为单一 Markdown 报告。"""
    all_groups = (
        ("单路基线", (dense_result, sparse_result, current_fusion)),
        ("dense_top_k", dense_top_results),
        ("sparse_top_k", sparse_top_results),
        ("rrf_k", rrf_k_results),
        ("固定 RRF 权重", weight_results),
        ("reranker candidate_pool", rerank_results),
    )
    sections = []
    for title, results in all_groups:
        sections.append(f"## {title}\n\n{_result_table(cases, results, current_fusion)}")

    pool10 = next(result for result in rerank_results if "pool=10" in result.name)
    best_rrf = max(
        rrf_k_results,
        key=lambda result: (
            result.metrics[4].complete_hit_count,
            result.metrics[4].gt_coverage,
            result.metrics[4].mrr,
        ),
    )
    best_weight = max(
        weight_results,
        key=lambda result: (
            result.metrics[4].complete_hit_count,
            result.metrics[4].gt_coverage,
            result.metrics[4].mrr,
        ),
    )
    best_rerank = max(
        rerank_results,
        key=lambda result: (
            result.metrics[4].complete_hit_count,
            result.metrics[4].gt_coverage,
            result.metrics[4].mrr,
        ),
    )
    multi_cases = tuple(case for case in cases if len(case.gt_chunk_ids) > 1)
    current_multi = evaluate_rankings(multi_cases, current_fusion.rankings, k=4)
    pool10_multi = evaluate_rankings(multi_cases, pool10.rankings, k=4)
    current_at4 = current_fusion.metrics[4]
    dense_at4 = dense_result.metrics[4]
    sparse_at4 = sparse_result.metrics[4]
    best_rrf_at4 = best_rrf.metrics[4]
    best_rerank_at4 = best_rerank.metrics[4]
    best_combination_key = max(
        (
            result.reranked.metrics[4].complete_hit_count,
            result.reranked.metrics[4].gt_coverage,
            result.reranked.metrics[4].mrr,
        )
        for result in rerank_combinations
    )
    best_combination_names = "`、`".join(
        result.name
        for result in rerank_combinations
        if (
            result.reranked.metrics[4].complete_hit_count,
            result.reranked.metrics[4].gt_coverage,
            result.reranked.metrics[4].mrr,
        )
        == best_combination_key
    )
    combination_results = tuple(result.reranked for result in rerank_combinations)
    best_lines = [
        f"- {title}：`{'`、`'.join(_best_names(results))}`"
        for title, results in all_groups
    ]
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    return (
        "# Retrieval 参数评估报告\n\n"
        "> 历史报告：本评估绑定旧单 query、top-4 口径，不代表当前多 query、"
        "top-5 检索与构包链路。\n\n"
        "> 本脚本只执行候选配置评估，不自动修改运行时默认参数。后续已从并列首位中采用 "
        "`dense_top_k=10`、`sparse_top_k=30`、`rrf_k=60`、`candidate_pool=20` "
        "和等权 RRF `1:1`，并同步至 `SemanticRetrievalConfig`。\n\n"
        "## 评估范围\n\n"
        f"- 数据：`rag/eval/eval_set.jsonl` 中 {len(cases)} 条 "
        "`legal_query + answer`。\n"
        "- 输入：只使用 `query_original`，不使用 `query_formal`。\n"
        "- 主指标：`complete_hit@4`；第二指标：`gt_coverage@4`。\n"
        "- 诊断指标：`any_hit@4`、`MRR@4`，并观察 `@5/@10`。\n"
        "- dense 与 BM25 各只执行一次 top-50；其余粗排配置在内存中重排。\n"
        "- 固定索引和模型，不重建索引，不比较动态权重。\n"
        f"- 生成时间：{generated_at}。\n\n"
        "## 固定环境\n\n"
        f"- embedding：`{DEFAULT_EMBEDDING_MODEL}`。\n"
        f"- reranker：`{DEFAULT_RERANKER_MODEL}`。\n"
        f"- 设备：`{device}`；reranker batch size：`{batch_size}`。\n"
        f"- dense 维度：{dense_metadata['vector_dimension']}；"
        f"最大长度：{dense_metadata['tokenizer_max_length']}；"
        f"窗口重叠：{dense_metadata['window_overlap_tokens']}。\n\n"
        "## 指标口径\n\n"
        "- `any_hit@k`：top-k 至少命中一条 GT 的题目比例。\n"
        "- `gt_coverage@k`：每题命中 GT 比例的宏平均，等同本文的法条级 Macro Recall@k；"
        "下文表格简称 Recall@k。\n"
        "- `complete_hit@k`：top-k 完整包含该题全部 GT 的题目比例。\n"
        "- `MRR@k`：首个 GT 排名倒数的宏平均，top-k 未命中记为 0。\n\n"
        + "\n\n".join(sections)
        + "\n\n## 单 GT 与多 GT\n\n"
        + _group_table(cases, (current_fusion, pool10))
        + "\n\n## 精排后的粗排参数组合验证\n\n"
        "第一轮固定 `dense_top_k=30`、`dense:BM25=1:1`，比较 "
        "`sparse_top_k={30, 50}`、`rrf_k={60, 10}`、`candidate_pool={10, 20}`；"
        "第二轮在同一结果表中加入 `dense_top_k={10, 20}`、`candidate_pool=20` 的 8 组。"
        "两轮合计 16 组，其中 pool=20 的 dense、sparse、rrf_k 交互矩阵共 12 组。"
        "候选池指标在各自 pool 深度计算，精排指标统一在 top-4 计算。\n\n"
        + _rerank_combination_table(rerank_combinations)
        + "\n\n### 精排组合的单 GT 与多 GT\n\n"
        + _group_table(cases, combination_results)
        + "\n\n### 两类失败题目\n\n"
        "这里先检查 GT 是否完整进入 candidate pool；只有已完整进入 pool、"
        "但精排后未完整保留在 top-4 的题目才计为精排降位。两类清单互斥。\n\n"
        + _rerank_failure_details(rerank_combinations)
        + "\n\n## 各组按既定指标排序的首位配置\n\n"
        + "\n".join(best_lines)
        + "\n\n该排序仅用于阅读报告，不触发参数修改。\n\n"
        "## 结果解读\n\n"
        f"- 当前等权 RRF 比 dense-only 多完整命中 "
        f"{current_at4.complete_hit_count - dense_at4.complete_hit_count} 题，"
        f"比 BM25-only 多 {current_at4.complete_hit_count - sparse_at4.complete_hit_count} 题；"
        "两路结果具有实际互补性。\n"
        f"- `rrf_k` 单变量比较中，`{best_rrf.name}` 比当前值多完整命中 "
        f"{best_rrf_at4.complete_hit_count - current_at4.complete_hit_count} 题，"
        f"GT coverage 增加 {(best_rrf_at4.gt_coverage - current_at4.gt_coverage) * 100:.2f} 个百分点。\n"
        f"- 固定权重比较的首位仍是 `{best_weight.name}`；本轮没有观察到非等权融合优于等权融合。\n"
        f"- reranker 候选池比较中，`{best_rerank.name}` 比未精排的当前等权 RRF 多完整命中 "
        f"{best_rerank_at4.complete_hit_count - current_at4.complete_hit_count} 题，"
        f"GT coverage 增加 {(best_rerank_at4.gt_coverage - current_at4.gt_coverage) * 100:.2f} 个百分点。\n"
        f"- 当前 `pool=10` 在多 GT 组完整命中为 {pool10_multi.complete_hit_count}/{pool10_multi.question_count}，"
        f"未精排等权 RRF 为 {current_multi.complete_hit_count}/{current_multi.question_count}；"
        "总体提升不能替代多 GT 分组检查。\n"
        f"- {len(rerank_combinations)} 组精排组合中，按 complete@4、Recall@4、"
        "MRR@4 依次排序的并列首位是 "
        f"`{best_combination_names}`；运行时从中采用的具体配置见报告顶部。\n"
        "- 前面的单变量比较用于筛选候选值，本节组合表用于验证精排后的参数交互；"
        "本报告仍未穷举连续参数范围。\n"
        "- 上述内容是实验观察，不是参数采用结论。\n\n"
        "## 运行耗时\n\n"
        f"- dense top-50：{timings['dense']:.2f} 秒。\n"
        f"- BM25 top-50：{timings['sparse']:.2f} 秒。\n"
        f"- reranker 评估候选并集：{timings['rerank']:.2f} 秒，共评分 "
        f"{rerank_unique_candidate_count} 个去重后的题目-候选对。\n"
        "- 上述 reranker 耗时来自多组配置的候选并集打分，不是任一单个生产配置的请求延迟。\n"
        "\n## 复现命令\n\n"
        "```powershell\n"
        "conda run --no-capture-output -n minimind python -m rag.eval.retrieval_parameter_report --batch-size 8\n"
        "```\n"
    )


def run_evaluation(*, eval_set, article_index, artifact_dir, output, device, batch_size):
    """执行一次完整的只读参数比较并写入 Markdown。"""
    repository = ArticleRepository.from_jsonl(article_index)
    cases = load_retrieval_cases(eval_set, repository)
    dense_searcher, sparse_searcher, reranker, actual_device, dense_metadata = (
        _load_components(
            repository=repository,
            artifact_dir=artifact_dir,
            device=device,
            batch_size=batch_size,
        )
    )
    dense_raw, dense_seconds = _collect_rankings(
        dense_searcher,
        cases,
        top_k=50,
        label="dense",
    )
    sparse_raw, sparse_seconds = _collect_rankings(
        sparse_searcher,
        cases,
        top_k=50,
        label="BM25",
    )
    dense_rankings = {
        case_id: tuple(item.chunk_id for item in results)
        for case_id, results in dense_raw.items()
    }
    sparse_rankings = {
        case_id: tuple(item.chunk_id for item in results)
        for case_id, results in sparse_raw.items()
    }
    dense_result = build_configuration_result("dense-only", cases, dense_rankings)
    sparse_result = build_configuration_result("BM25-only", cases, sparse_rankings)

    def fusion_result(name, **kwargs):
        rankings = build_fusion_rankings(cases, dense_raw, sparse_raw, **kwargs)
        return build_configuration_result(name, cases, rankings)

    current_fusion = fusion_result("当前等权 RRF (30/30, k=60)")
    dense_top_results = tuple(
        fusion_result(f"dense_top_k={value}", dense_top_k=value)
        for value in (10, 20, 30, 50)
    )
    sparse_top_results = tuple(
        fusion_result(f"sparse_top_k={value}", sparse_top_k=value)
        for value in (10, 20, 30, 50)
    )
    rrf_k_results = tuple(
        fusion_result(f"rrf_k={value}", rrf_k=value)
        for value in (10, 30, 60, 100)
    )
    weight_results = tuple(
        fusion_result(
            f"dense:BM25={dense_weight}:{sparse_weight}",
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        for dense_weight, sparse_weight in ((1, 0), (3, 1), (1, 1), (1, 3), (0, 1))
    )

    current_pool_names = {
        pool: f"current-pool-{pool}" for pool in (5, 10, 20, 30)
    }
    combination_specs = (
        tuple(
            (30, sparse_top_k, rrf_k, pool)
            for sparse_top_k in (30, 50)
            for rrf_k in (60, 10)
            for pool in (10, 20)
        )
        + tuple(
            (dense_top_k, sparse_top_k, rrf_k, 20)
            for dense_top_k in (10, 20)
            for sparse_top_k in (30, 50)
            for rrf_k in (60, 10)
        )
    )
    coarse_specs = tuple(dict.fromkeys(spec[:3] for spec in combination_specs))
    coarse_combinations = {
        spec: fusion_result(
            (
                f"组合粗排 dense={spec[0]}, sparse={spec[1]}, "
                f"rrf_k={spec[2]}"
            ),
            dense_top_k=spec[0],
            sparse_top_k=spec[1],
            rrf_k=spec[2],
        )
        for spec in coarse_specs
    }
    combination_names = {
        spec: f"combination-{index}"
        for index, spec in enumerate(combination_specs, start=1)
    }
    requirements = {
        name: (current_fusion.rankings, pool)
        for pool, name in current_pool_names.items()
    }
    requirements.update(
        {
            combination_names[spec]: (
                coarse_combinations[spec[:3]].rankings,
                spec[3],
            )
            for spec in combination_specs
        }
    )
    reranked, rerank_seconds, rerank_unique_candidate_count = _rerank_requirements(
        cases,
        requirements,
        repository,
        reranker,
    )
    rerank_results = tuple(
        build_configuration_result(
            f"当前等权 RRF + reranker pool={pool}",
            cases,
            reranked[current_pool_names[pool]],
        )
        for pool in (5, 10, 20, 30)
    )
    rerank_combinations = []
    for dense_top_k, sparse_top_k, rrf_k, pool in combination_specs:
        spec = (dense_top_k, sparse_top_k, rrf_k, pool)
        coarse = coarse_combinations[spec[:3]]
        reranked_rankings = reranked[combination_names[spec]]
        reranked_result = build_configuration_result(
            (
                f"dense={dense_top_k}, sparse={sparse_top_k}, "
                f"rrf_k={rrf_k}, pool={pool}"
            ),
            cases,
            reranked_rankings,
        )
        missing_from_pool, demoted_after_rerank = classify_rerank_failures(
            cases,
            coarse.rankings,
            reranked_rankings,
            pool=pool,
        )
        rerank_combinations.append(
            RerankCombinationResult(
                dense_top_k=dense_top_k,
                sparse_top_k=sparse_top_k,
                rrf_k=rrf_k,
                candidate_pool=pool,
                pool_metrics=evaluate_rankings(cases, coarse.rankings, k=pool),
                reranked=reranked_result,
                missing_from_pool=missing_from_pool,
                demoted_after_rerank=demoted_after_rerank,
            )
        )
    rerank_combinations = tuple(rerank_combinations)
    report = render_report(
        cases=cases,
        dense_result=dense_result,
        sparse_result=sparse_result,
        current_fusion=current_fusion,
        dense_top_results=dense_top_results,
        sparse_top_results=sparse_top_results,
        rrf_k_results=rrf_k_results,
        weight_results=weight_results,
        rerank_results=rerank_results,
        rerank_combinations=rerank_combinations,
        rerank_unique_candidate_count=rerank_unique_candidate_count,
        timings={
            "dense": dense_seconds,
            "sparse": sparse_seconds,
            "rerank": rerank_seconds,
        },
        device=actual_device,
        batch_size=batch_size,
        dense_metadata=dense_metadata,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"报告已写入: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="生成 retrieval 参数评估 Markdown")
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument("--article-index", default=str(DEFAULT_ARTICLE_INDEX))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size 必须是正整数")
    run_evaluation(
        eval_set=args.eval_set,
        article_index=args.article_index,
        artifact_dir=args.artifact_dir,
        output=args.output,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
