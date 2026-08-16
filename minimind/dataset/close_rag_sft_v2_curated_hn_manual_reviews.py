"""闭合 curated HN 人工双审草稿，不投影或物化正式训练 HN。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .build_rag_sft_v2_curated_hn_reuse_review import (
    HASH_FILENAME, MANIFEST_FILENAME, _bound_path, _identity, _load_json,
    _load_jsonl, _payload_identity, _sha256, _serialize_jsonl, _stable_visible,
    _verify_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUEUE = PROJECT_ROOT / "minimind/dataset/RAG-SFT/review/v2/curated-hn-manual-review-queue-v1-20260814"
DEFAULT_DRAFTS = PROJECT_ROOT / "minimind/dataset/RAG-SFT/review/v2/curated-hn-manual-drafts-v1-20260814"
DEFAULT_OUTPUT = PROJECT_ROOT / "minimind/dataset/RAG-SFT/review/v2/curated-hn-manual-semantic-review-v1-20260814"

class CuratedHnManualReviewError(RuntimeError):
    """人工双审草稿的身份、覆盖或准入规则未闭合。"""

def _unique(rows: list[dict[str, Any]], label: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        q, c, r = row.get("query_id"), row.get("chunk_id"), row.get("candidate_rank")
        if not isinstance(q, str) or not isinstance(c, str) or type(r) is not int:
            raise CuratedHnManualReviewError(f"{label}身份字段无效")
        key = (q, c, r)
        if key in result:
            raise CuratedHnManualReviewError(f"{label}存在重复候选: {q}")
        result[key] = row
    return result

def close_rag_sft_v2_curated_hn_manual_reviews(*, queue_root: Path, drafts_root: Path, output_dir: Path, suffix: str = "") -> dict[str, object]:
    queue_root, drafts_root, output_dir = Path(queue_root).resolve(), Path(drafts_root).resolve(), Path(output_dir).resolve()
    partial = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial.exists():
        raise CuratedHnManualReviewError(f"输出目录或临时目录已存在: {output_dir}")
    queue_manifest_path = queue_root / MANIFEST_FILENAME
    queue_sha = _verify_manifest(queue_manifest_path)
    queue_manifest = _load_json(queue_manifest_path, "人工队列 manifest")
    if queue_manifest.get("pipeline") != "rag_sft_v2_curated_hn_manual_review_queue" or queue_manifest.get("policy", {}).get("training_ready") is not False:
        raise CuratedHnManualReviewError("人工队列状态无效")
    queue_path = _bound_path(queue_manifest.get("output", {}).get("manual_review_queue"), "人工队列")
    queue = {row.get("query_id"): row for row in _load_jsonl(queue_path, "人工队列")}
    if len(queue) != 381 or any(not isinstance(key, str) for key in queue):
        raise CuratedHnManualReviewError("人工队列 query 身份不闭合")
    pairs = [("a", "c", f"agent-a{suffix}-adversarial.jsonl", f"agent-c{suffix}-legal-a.jsonl"), ("b", "a", f"agent-b{suffix}-adversarial.jsonl", f"agent-a{suffix}-legal-b.jsonl"), ("c", "b", f"agent-c{suffix}-adversarial.jsonl", f"agent-b{suffix}-legal-c.jsonl")]
    adversarial: list[dict[str, object]] = []; legal: list[dict[str, object]] = []; adjudication: list[dict[str, object]] = []; variants: list[dict[str, object]] = []; ledger: list[dict[str, object]] = []; input_drafts=[]
    seen_queries:set[str]=set()
    for partition, legal_partition, adv_name, legal_name in pairs:
        adv_path, legal_path = drafts_root / adv_name, drafts_root / legal_name
        adv = _unique(_load_jsonl(adv_path, adv_name), adv_name); laws = _unique(_load_jsonl(legal_path, legal_name), legal_name)
        if set(adv) != set(laws): raise CuratedHnManualReviewError(f"{adv_name}与{legal_name}覆盖不一致")
        input_drafts.extend([_identity(adv_path, len(adv)), _identity(legal_path, len(laws))])
        for key, a in adv.items():
            q,c,r=key; l=laws[key]; item=queue.get(q)
            candidates=item.get("candidate_articles") if isinstance(item,dict) else None
            match = next((x for x in candidates or [] if x.get("chunk_id")==c and x.get("rank")==r), None)
            if not isinstance(item,dict) or item.get("review_partition") != partition or not isinstance(match,dict) or a.get("adversarial_label") != "hard_negative":
                raise CuratedHnManualReviewError(f"草稿与人工队列身份或对抗标签不闭合: {q}")
            support=l.get("legal_support")
            if support not in {"direct","partial","alternative","none","uncertain"}: raise CuratedHnManualReviewError("法律支持标签无效")
            if q in seen_queries: raise CuratedHnManualReviewError(f"人工草稿跨组 query 重复: {q}")
            seen_queries.add(q); variant_id=f"{q}:hn:curated:manual-v1"; review_id=f"{variant_id}:{c}"
            adversarial.append({"review_id":review_id,"query_id":q,"variant_id":variant_id,"chunk_id":c,"adversarial_label":"hard_negative","reason":a.get("reason"),"reviewer":a.get("reviewer")})
            legal.append({"review_id":review_id,"query_id":q,"variant_id":variant_id,"chunk_id":c,"legal_support":support,"reason":l.get("reason"),"reviewer":l.get("reviewer")})
            approved=support=="none"; adjudication.append({"review_id":review_id,"query_id":q,"variant_id":variant_id,"chunk_id":c,"adversarial_label":"hard_negative","legal_support":support,"decision":"approve_evidence" if approved else "exclude_evidence","decision_reason":"同时满足 hard_negative 与 none，准入。" if approved else "法律支持不为 none，保守排除。","adjudicator":"curated_manual_adjudication_v1"})
            ledger.append({"review_id":review_id,"query_id":q,"chunk_id":c,"candidate_rank":r,"queue_path":str(queue_path),"queue_sha256":_sha256(queue_path),"article_source":"rag/chunk/article_index.jsonl","article_law_name":match.get("law_name"),"article_no":match.get("article_no"),"adversarial_draft":adv_name,"legal_draft":legal_name})
            if approved:
                required=[x.get("chunk_id") for x in item.get("required_gt",[])];
                if not required or any(not isinstance(x,str) for x in required): raise CuratedHnManualReviewError(f"required GT无效: {q}")
                variants.append({"variant_id":variant_id,"query_id":q,"visible_chunk_ids":_stable_visible(required,q,c),"source":"curated","retrieval_identity":f"curated:manual:queue-sha256:{_sha256(queue_path)}:rank:{r}","non_gt_labels":[{"chunk_id":c,"label":"hard_negative"}],"review_decision":"approved"})
    payloads={"curated-hn-variants.jsonl":_serialize_jsonl(variants),"adversarial.jsonl":_serialize_jsonl(adversarial),"legal-support.jsonl":_serialize_jsonl(legal),"adjudication.jsonl":_serialize_jsonl(adjudication),"source-evidence-ledger.jsonl":_serialize_jsonl(ledger)}
    manifest={"pipeline":"rag_sft_v2_curated_hn_manual_semantic_review","release_status":"curated_hn_manual_semantic_review_closed","inputs":{"queue_manifest":{**_identity(queue_manifest_path),"manifest_sha256":queue_sha},"queue":_identity(queue_path,381),"independent_drafts":input_drafts},"records":{"reviewed":len(adjudication),"approved":len(variants),"excluded":len(adjudication)-len(variants),"adversarial_labels":{"hard_negative":len(adversarial)},"legal_support":{label:sum(x["legal_support"]==label for x in legal) for label in ("direct","partial","alternative","none","uncertain")}},"policy":{"source_candidates_modified":False,"formal_hn_materialized":False,"context_length_audit":"deferred","training_ready":False},"validation":{"strict_utf8_jsonl":True,"independent_double_review_covered":True,"admit_rule":"adversarial_label == hard_negative and legal_support == none","one_query_one_variant":True},"output":{name:_payload_identity(output_dir/name,payload,len(payload.splitlines())) for name,payload in payloads.items()},"complete":True}
    payloads[MANIFEST_FILENAME]=json.dumps(manifest,ensure_ascii=False,indent=2)+"\n"; payloads["README.md"]="# RAG-SFT v2 curated HN 人工双审闭合\n\n本目录闭合独立对抗与法律支持审阅；仅 `hard_negative + none` 准入。未物化正式 HN，`training_ready=false`，上下文长度审核延后。\n"
    try:
        partial.mkdir(parents=True,exist_ok=False)
        for name,payload in payloads.items(): (partial/name).write_text(payload,encoding="utf-8",newline="\n")
        (partial/HASH_FILENAME).write_text("".join(f"{_sha256(partial/name)}  {name}\n" for name in payloads),encoding="ascii",newline="\n"); partial.replace(output_dir)
    except OSError as error: raise CuratedHnManualReviewError("无法原子发布人工双审闭合资产；已保留临时目录") from error
    return manifest

def main()->None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--queue-root",type=Path,default=DEFAULT_QUEUE); parser.add_argument("--drafts-root",type=Path,default=DEFAULT_DRAFTS); parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT); parser.add_argument("--suffix",default=""); args=parser.parse_args()
    try: manifest=close_rag_sft_v2_curated_hn_manual_reviews(queue_root=args.queue_root,drafts_root=args.drafts_root,output_dir=args.output_dir,suffix=args.suffix)
    except (CuratedHnManualReviewError,OSError,ValueError,TypeError) as error: parser.error(str(error))
    print(f"RAG_SFT_V2_CURATED_MANUAL_REVIEW_OK approved={manifest['records']['approved']} training_ready=false")
if __name__=="__main__": main()
