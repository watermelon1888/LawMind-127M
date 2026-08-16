"""
eval_set 生成/校验共享的纯函数库。
generate（rebuild_eval_set.py）和 verify（verify_eval_set.py）都从这里导入，
确保"条号是否存在"等判断逻辑只有一份实现 —— 这是 Q150 幻觉 bug 的根因修复：
之前生成脚本手写猜测、校验脚本又不做真实性核验，两边各自为政导致错误未被拦截。
"""
import re

CITATION_PAT = re.compile(r'《([^》]+)》第(\d+)条(?:之([一二三四五六七八九十]+))?')

_CN_NUM = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}


def extract_citation(text):
    """
    从自然语言文本中提取第一个 (law_name, article_no) 引用。
    输入: "《中华人民共和国刑法》第264条的内容是什么？"
    输出: ("中华人民共和国刑法", "264")
    输入: "《中华人民共和国刑法》第133条之一是什么" → ("中华人民共和国刑法", "133之一")
    输入: 无引用文本 → None
    """
    m = CITATION_PAT.search(text)
    if not m:
        return None
    law_name, main_no, suffix_cn = m.group(1), m.group(2), m.group(3)
    article_no = main_no if not suffix_cn else f"{main_no}之{suffix_cn}"
    return (law_name, article_no)


def build_citation(law_name, article_no):
    """构造标准引用格式字符串，如 '《中华人民共和国刑法》第133条之一'"""
    if "之" in article_no:
        main, suffix = article_no.split("之", 1)
        return f"《{law_name}》第{main}条之{suffix}"
    return f"《{law_name}》第{article_no}条"


def build_citation_multi(articles):
    """
    构造多条法条的复合引用格式。
    articles: [(law_name, article_no), ...]
    单条 → "《刑法》第264条"
    多条 → "《刑法》第264条；《治安管理处罚法》第58条"
    """
    return "；".join(build_citation(law, ano) for law, ano in articles)


def build_gt_answer(articles):
    """
    构造 gt_answer：多条法条时用双换行拼接全部条文。
    articles: [article_dict, ...]（含 law_name, article_no, content 字段）
    """
    return "\n\n".join(a["content"] for a in articles)


def build_indexes(articles):
    """
    从 article 列表构建三个索引:
      lookup: (law_name, article_no) -> article
      by_law: law_name -> [article, ...]
      chunk_lookup: chunk_id -> article
    """
    lookup = {}
    by_law = {}
    chunk_lookup = {}
    for a in articles:
        lookup[(a["law_name"], a["article_no"])] = a
        by_law.setdefault(a["law_name"], []).append(a)
        chunk_lookup[a["chunk_id"]] = a
    return lookup, by_law, chunk_lookup


def resolve_law_name(by_law, law_name):
    """
    把 query 里出现的法律名解析为语料库中的正式全名。
    有些法律语料库存的是带版本后缀的正式名（如"中华人民共和国宪法（2018年修正文本）"），
    而 query 里常用不带后缀的简称。规则：
      1. 精确匹配 → 直接返回
      2. 唯一前缀匹配（语料库中恰好一个法律名以 law_name 开头）→ 返回该正式名
      3. 无匹配或前缀匹配有歧义（多个候选）→ 返回 None，交给上层判定为"库外法律"
    """
    if law_name in by_law:
        return law_name
    candidates = [name for name in by_law if name.startswith(law_name)]
    if len(candidates) == 1:
        return candidates[0]
    return None


def check_law_exists(by_law, law_name):
    """法律名是否在语料库中确有记录（至少 1 条）"""
    return law_name in by_law and len(by_law[law_name]) > 0


def check_article_missing(by_law, law_name, article_no):
    """
    核验某条号在指定法律下是否不存在。
    前提: 法律必须存在，否则抛 ValueError（law 都不存在时"条号不存在"这个判断没有意义，
    应该走"法律不存在"这条完全不同的错误路径，而不是被这个函数悄悄吞掉）。
    """
    if not check_law_exists(by_law, law_name):
        raise ValueError(f"法律 '{law_name}' 在语料库中不存在，无法核验条号")
    real_article_nos = {a["article_no"] for a in by_law[law_name]}
    return article_no not in real_article_nos


def build_missing_article_refusal(by_law, law_name, article_no):
    """
    生成"条号不存在"类不可答条目的 refusal_golden 话术，话术中的真实条数从语料库统计得出。
    核心防幻觉断言：若法律不存在，或条号其实真实存在（说明这不是"条号不存在"场景，
    而是 Q150 式误判），一律抛 ValueError 中断生成，绝不静默产出幻觉话术。
    """
    if not check_law_exists(by_law, law_name):
        raise ValueError(f"法律 '{law_name}' 在语料库中不存在，不能用于'条号不存在'类不可答条目")
    if not check_article_missing(by_law, law_name, article_no):
        raise ValueError(
            f"'{law_name}' 第{article_no}条实际存在于语料库中——这是 Q150 式误判，请检查 query 是否选错了条号")
    real_count = len(by_law[law_name])
    cite = build_citation(law_name, article_no)
    return f"{cite}不存在，《{law_name}》共{real_count}条，请您核实条号是否正确。"


def resolve_exact_lookup(lookup, law_name, article_no):
    """精确查询: (law_name, article_no) 必须能在 lookup 中查到，查不到抛 KeyError 交给上层处理"""
    key = (law_name, article_no)
    if key not in lookup:
        raise KeyError(f"'{law_name}' 第{article_no}条 不存在于语料库中")
    return lookup[key]


def pick_cross_law_articles(ranked_articles, max_articles=2):
    """
    从已按相关性排序的候选法条中挑选跨部门 gt_articles。
    优先选取第一条 + 第一个不同部门的候选；若没有其他部门，退化为按原排名截取前 N 条。
    """
    if not ranked_articles:
        return []
    picked = [ranked_articles[0]]
    first_dept = ranked_articles[0]["department"]
    for art in ranked_articles[1:]:
        if len(picked) >= max_articles:
            break
        if art["department"] != first_dept:
            picked.append(art)
    if len(picked) < max_articles:
        for art in ranked_articles[1:]:
            if len(picked) >= max_articles:
                break
            if art not in picked:
                picked.append(art)
    return picked[:max_articles]


def apply_chain_hints(entries):
    """
    根据 chain_hint（如 'CHAIN_1_R1'/'CHAIN_1_R2'）把同链条目两两配对，
    给 R1 条目写入 multi_turn_followup = R2 的 id。不修改没有 chain_hint 的条目。
    """
    by_chain = {}
    for e in entries:
        hint = e.get("chain_hint")
        if not hint:
            continue
        chain_id = hint.rsplit("_", 1)[0]
        by_chain.setdefault(chain_id, {})[hint] = e

    for chain_id, rounds in by_chain.items():
        r1_key = f"{chain_id}_R1"
        r2_key = f"{chain_id}_R2"
        if r1_key in rounds and r2_key in rounds:
            rounds[r1_key]["multi_turn_followup"] = rounds[r2_key]["id"]
