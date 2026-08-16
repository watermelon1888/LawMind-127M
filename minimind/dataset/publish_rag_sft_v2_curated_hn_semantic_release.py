"""发布非训练的 RAG-SFT v2 curated HN 语义/身份冻结 release。"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
from rag.knowledge import ArticleRepository
from .build_rag_sft_v2_curated_hn_reuse_review import _identity,_load_json,_load_jsonl,_bound_path,_payload_identity,_serialize_jsonl,_sha256,_verify_manifest,_canonical_from_clean
from .rag_sft_v2_projection import project_rag_sft_v2_record

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'minimind/dataset/RAG-SFT/releases/v2/rag-sft-v2-training-release-619-v1-20260813'
REVIEW=ROOT/'minimind/dataset/RAG-SFT/review/v2'
INDEX=ROOT/'rag/chunk/article_index.jsonl'
OUT=ROOT/'minimind/dataset/RAG-SFT/releases/v2/rag-sft-v2-curated-hn-semantic-release-779-v1-20260814'
class Error(RuntimeError): pass
def publish(*,base:Path,review:Path,index:Path,out:Path)->dict[str,object]:
 base,review,index,out=map(lambda x:Path(x).resolve(),(base,review,index,out)); partial=out.with_name(out.name+'.partial')
 if out.exists() or partial.exists(): raise Error('输出目录或临时目录已存在')
 bm=base/'manifest.json'; bsha=_verify_manifest(bm); mb=_load_json(bm,'基线 manifest'); bp=_bound_path(mb['output']['training_candidate'],'基线 candidate'); rows=_load_jsonl(bp,'基线 candidate')
 if len(rows)!=619 or mb.get('records',{}).get('clean')!=549 or mb.get('records',{}).get('hard_negative')!=70: raise Error('619 基线身份不闭合')
 clean={r['query_id']:r for r in rows if r.get('variant')=='clean'}; existing={r['query_id'] for r in rows if r.get('variant')=='hard_negative'}
 roots=[review/'curated-hn-reuse-review-v1-20260814',*[review/f'curated-hn-manual-semantic-review-v{i}-20260814' for i in range(1,6)]]
 variants=[]; inputs=[]
 for root in roots:
  mp=root/'manifest.json'; sha=_verify_manifest(mp); m=_load_json(mp,'审阅 manifest')
  if m.get('policy',{}).get('training_ready') is not False or m.get('policy',{}).get('formal_hn_materialized') is not False: raise Error('审阅资产状态无效')
  output=m.get('output',{}); meta=output.get('curated_hn_variants') or output.get('curated-hn-variants.jsonl')
  vp=_bound_path(meta,'curated variants'); vs=_load_jsonl(vp,'curated variants'); variants+=vs; inputs.append({'manifest':{**_identity(mp),'manifest_sha256':sha},'variants':_identity(vp,len(vs))})
 if len(variants)!=160: raise Error(f'预期160条 curated，实际{len(variants)}')
 repo=ArticleRepository.from_jsonl(index); projected=[]; ledger=[]; seen=set()
 for v in variants:
  q=v.get('query_id')
  if not isinstance(q,str) or q in seen or q in existing or q not in clean: raise Error('curated query 身份重叠或缺少 clean')
  seen.add(q); canonical=_canonical_from_clean(clean[q]); arts={cid:repo.get_by_chunk_id(cid) for cid in v['visible_chunk_ids']}; p=project_rag_sft_v2_record(canonical,arts,hn_variant=v)
  record={'id':p.record_id,'query_id':p.query_id,'variant':p.variant,'query_original':canonical['query_original'],'visible_chunk_ids':list(p.visible_chunk_ids),'required_chunk_ids':list(p.required_chunk_ids),'citations':list(p.citations),'conversations':p.training_conversations()}; projected.append(record); ledger.append({'id':p.record_id,'query_id':q,'variant':'hard_negative','source':'curated','variant_id':v['variant_id'],'record_sha256':hashlib.sha256(json.dumps(record,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()})
 candidate=[*rows,*projected]
 cp=_serialize_jsonl(candidate); vp=_serialize_jsonl(variants); lp=_serialize_jsonl(ledger)
 manifest={'pipeline':'rag_sft_v2_curated_hn_semantic_release','release_status':'semantic_identity_freeze_only','inputs':{'base_manifest':{**_identity(bm),'manifest_sha256':bsha},'base_candidate':_identity(bp,619),'article_index':_identity(index),'curated_reviews':inputs},'records':{'total':len(candidate),'clean':549,'retrieved_hn':70,'curated_hn':160,'hard_negative':230,'unique_queries':549,'paired_hn_queries':230},'policy':{'source_assets_modified':False,'formal_hn_materialized':False,'formal_training_candidate_emitted':False,'context_length_audit':'deferred','training_ready':False},'validation':{'strict_utf8_jsonl':True,'query_unique_per_hn':True,'all_curated_have_clean':True,'required_gt_and_summary_preserved':True,'no_tokenizer_or_context_audit_run':True,'quarantined_batch7_legacy_not_referenced':True},'output':{'semantic_freeze_candidate':_payload_identity(out/'semantic-freeze-candidate.jsonl',cp,len(candidate)),'curated_hn_variants':_payload_identity(out/'curated-hn-variants.jsonl',vp,len(variants)),'identity_ledger':_payload_identity(out/'identity-ledger.jsonl',lp,len(ledger))},'complete':True}
 mp=json.dumps(manifest,ensure_ascii=False,indent=2)+'\n'; readme='# RAG-SFT v2 curated HN 语义/身份冻结 release\n\n本目录不是正式训练入口；未做上下文长度审核，`formal_hn_materialized=false`，`training_ready=false`。\n'; payload={'semantic-freeze-candidate.jsonl':cp,'curated-hn-variants.jsonl':vp,'identity-ledger.jsonl':lp,'manifest.json':mp,'README.md':readme}
 partial.mkdir(parents=True)
 for n,p in payload.items():(partial/n).write_text(p,encoding='utf-8',newline='\n')
 (partial/'manifest.sha256').write_text(''.join(f'{_sha256(partial/n)}  {n}\n' for n in payload),encoding='ascii',newline='\n'); partial.replace(out); return manifest
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=BASE);p.add_argument('--review',type=Path,default=REVIEW);p.add_argument('--index',type=Path,default=INDEX);p.add_argument('--output-dir',type=Path,default=OUT);a=p.parse_args();m=publish(base=a.base,review=a.review,index=a.index,out=a.output_dir);print(f"RAG_SFT_V2_CURATED_SEMANTIC_RELEASE_OK total={m['records']['total']} training_ready=false")
if __name__=='__main__':main()
