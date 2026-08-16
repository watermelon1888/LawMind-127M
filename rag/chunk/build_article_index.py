"""
Step 1: 全量法条提取脚本（v3 — 完整 11 字段 schema）
遍历 rag/Chinese-Laws/ 下所有 .docx，逐条提取法条元数据 → article_index.jsonl

Schema 定义见 SCHEMA.md。每个 chunk 包含：
  chunk_id, law_name, article_no, article_no_sort_key,
  content, token_count, char_count,
  department, effective_date, hierarchy

"""
import os, re, json
from datetime import datetime
from docx2txt import process as docx2txt_process

#定位输入和输出目录
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAW_DIR = os.path.join(BASE, "Chinese-Laws")
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "article_index.jsonl")#产物路径

# ============================================================
# 基础工具函数
# ============================================================

# 中文数字 → 阿拉伯数字转换表
CN_NUM = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
    '百': 100, '千': 1000, '零': 0,
}

# 合法的中国法律部门枚举
VALID_DEPARTMENTS = {
    '刑法', '民法商法', '行政法', '经济法',
    '社会法', '生态环境法', '诉讼与非诉讼程序法', '宪法及宪法相关法',
}


def cn2arabic(s):
    """
    将中文数字字符串转为阿拉伯数字。
    输入: "二百六十四" → 输出: "264"
    输入: "十三" → 输出: "13"
    输入: "一千零一" → 输出: "1001"
    输入: "abc"（含非中文数字字符） → 输出: "abc"（原样返回）
    输入: "" → 输出: ""

    使用正向逐字累加算法：数字（一~九）暂存到 current，
    遇到量级标记（十/百/千）时 current × 量级 → 累加到 result。
    """
    if not s:
        return ""
    result = 0
    current = 0
    for ch in s:
        if ch not in CN_NUM:
            return s
        val = CN_NUM[ch]
        if val >= 10:
            if current == 0:
                current = 1
            current *= val
            result += current
            current = 0
        else:
            current = val
    result += current
    return str(result)

#===========article_no_sort_key===============
def parse_sort_key(article_no):
    """
    将规范化的 article_no 字符串转为可排序的整数数组。
    输入: "264" → 输出: [264]
    输入: "133之一" → 输出: [133, 1]
    输入: "230之二" → 输出: [230, 2]

    用途：字符串排序下 "2" > "10"，数组字典序消除此问题。
    "之一" 紧跟在主条号之后（[133, 1] > [133] > [132]）。
    """
    if '之' in article_no:
        main_str, suffix_cn = article_no.split('之', 1)#分割
        return [int(main_str), CN_NUM.get(suffix_cn, 99)]#如果后缀不在映射表中（理论上不会发生），默认值 99
    return [int(article_no)]

#===========chunk_id ===============
def build_chunk_id(law_name, article_no):
    """
    构造全局唯一 chunk 标识符。
    输入: ("中华人民共和国刑法", "264") → 输出: "中华人民共和国刑法#264"
    输入: ("中华人民共和国刑法", "133之一") → 输出: "中华人民共和国刑法#133之一"

    格式: {law_name}#{article_no}，分隔符 # 在中文法律名称中不存在，安全。
    """
    return f"{law_name}#{article_no}"

#===========effective_date ===============
def extract_effective_date(fname):
    """
    从文件名中提取施行日期。
    输入: "中华人民共和国刑法_20201226.docx" → 输出: "2020-12-26"
    输入: "unknown.docx"（无法解析） → 输出: None

    文件名格式约定: {法律名}_{YYYYMMDD}.docx
    """
    m = re.search(r'_(\d{8})\.docx$', fname)
    if not m:
        return None
    raw = m.group(1)
    dt = datetime.strptime(raw, '%Y%m%d')
    return dt.strftime('%Y-%m-%d')


# ============================================================
# 正文预处理
# ============================================================

# 法条匹配: 第X条 / 第X条之一 / 第X条之Y
ARTICLE_PAT = re.compile(
    r'(?:^|\n)\s*第([一二三四五六七八九十百千零]+)条(之([一二三四五六七八九十]+))?\s*\n?'
)


def pre_clean_text(text):
    """
    预处理法律全文：移除目录、页码残留，规范化空白。
    不剥离编/章/节层级标题——它们需要保留给 extract_chapters 定位，
    且它们位于法条之间的间隙，不会混入法条 content。
    输入: docx 原始全文
    输出: 清洗后的正文
    """
    # 1. 目录检测与裁切
    toc_end = 0
    lines = text.split('\n')
    toc_line_count = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'第[一二三四五六七八九十百千零]+[章节编]', stripped) and len(stripped) < 40:
            toc_line_count += 1
        elif toc_line_count > 3 and stripped and not re.match(r'第[一二三四五六七八九十百千零]+[章节编]', stripped):
            if toc_line_count > 5:
                toc_end = i
            toc_line_count = 0
        else:
            if toc_line_count <= 5:
                toc_line_count = 0
    if toc_end > 0:
        text = '\n'.join(lines[toc_end:])

    # 2. 去 .docx 页码残留
    text = re.sub(r'[\s\n]*[－-]\d+[－-][\s\n]*', '\n', text)

    # 3. 规范化空白
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# 层级标题行模式 — 用于清理法条 content 尾部的残留标题
HIERARCHY_HEADER = re.compile(
    r'\n+[\s　]*第[一二三四五六七八九十百千零]+(?:编|分编|章|节|款)[\s　]*[^\n]*$'
)


def clean_article_content(content):
    """
    清洗单条法条内容。
    1. 去掉尾部页码残留（－1－）
    2. 去掉尾部残留的层级标题行 — 法条之间的编/章/节标题会粘到上一法条末尾
    """
    content = re.sub(r'[\s]*[－-]\d+[－-][\s]*$', '', content)
    content = HIERARCHY_HEADER.sub('', content)
    return content.strip()


def is_deleted_article(content):
    """检测是否为已删除的占位条文（如刑法修正案标记'（删去）'）。"""
    stripped = content.strip()
    return stripped in ('（删去）', '(删去)', '删除', '（删除）') or len(stripped) <= 3


# ============================================================
# 层级提取与分配
# ============================================================

def _norm_title(raw):
    """
    规范化层级标题：压缩全角/半角空格、去首尾空白。
    输入: "刑法的任务、基本原则和适用范围" 或 "刑　法　的　任　务"
    输出: "刑法的任务、基本原则和适用范围"
    """
    return re.sub(r'[\s　]+', '', raw).strip()


def extract_chapters(text):
    """
    扫描全文提取所有层级标题（编/分编/章/节）及其文本位置。
    输入: 法律全文
    输出: [{"type": "编", "num": "2", "title": "第二编 分则", "pos": 1500}, ...]
          按 pos（在原文中的字符偏移量）升序排列。

    支持 4 种层级。分编仅民法典含有（311 部中仅此 1 部），
    其余法律无此层级时不会出现在返回列表中。
    """
    chapters = []

    # 编（如"第一编 总则"）— 行首锚定，防止法条正文中"本法第二编"类引用被误抓
    for m in re.finditer(r'(?m)^第([一二三四五六七八九十百千零]+)编\s*[　 ]?\s*([^\n]{2,30})', text):
        chapters.append({"type": "编", "num": cn2arabic(m.group(1)),
                         "title": f"第{cn2arabic(m.group(1))}编 {_norm_title(m.group(2))}",
                         "pos": m.start()})

    # 分编（如"第一分编 通则"，仅民法典使用）
    for m in re.finditer(r'(?m)^第([一二三四五六七八九十百千零]+)分编\s*[　 ]?\s*([^\n]{2,40})', text):
        chapters.append({"type": "分编", "num": cn2arabic(m.group(1)),
                         "title": f"第{cn2arabic(m.group(1))}分编 {_norm_title(m.group(2))}",
                         "pos": m.start()})

    # 章（如"第一章 基本规定"）
    for m in re.finditer(r'(?m)^第([一二三四五六七八九十百千零]+)章\s*[　 ]?\s*([^\n]{2,40})', text):
        chapters.append({"type": "章", "num": cn2arabic(m.group(1)),
                         "title": f"第{cn2arabic(m.group(1))}章 {_norm_title(m.group(2))}",
                         "pos": m.start()})

    # 节（如"第一节 不动产登记"）
    for m in re.finditer(r'(?m)^第([一二三四五六七八九十百千零]+)节\s*[　 ]?\s*([^\n]{2,40})', text):
        chapters.append({"type": "节", "num": cn2arabic(m.group(1)),
                         "title": f"第{cn2arabic(m.group(1))}节 {_norm_title(m.group(2))}",
                         "pos": m.start()})

    chapters.sort(key=lambda x: x['pos'])
    return chapters


def assign_hierarchy(article_start_pos, chapters):
    """
    根据法条在文本中的起始位置，判定其所属的层级栈。
    输入: article_start_pos (int) — 法条"第X条"在文本中的字符偏移
          chapters (list) — extract_chapters() 的输出，按 pos 升序
    输出: dict — 该位置当前的层级栈。

    采用栈式更新：遇到高级别标题（编）时清空下级（章、节）；
    遇到章标题时清空节。确保不会把前一个章的节归属到当前章。

    例如：第二编 → 第三章 → 第八节 → 第四章 → 第五章 → 第二百六十四条
      → {"编": "第二编 分则", "章": "第五章 侵犯财产罪"}（第八节属于第三章，已被清掉）
    """
    hier = {}
    for ch in chapters:
        if ch['pos'] >= article_start_pos:
            break
        t = ch['type']
        if t == '编':
            hier.pop('分编', None)
            hier.pop('章', None)
            hier.pop('节', None)
        elif t == '分编':
            hier.pop('章', None)
            hier.pop('节', None)
        elif t == '章':
            hier.pop('节', None)
        hier[t] = ch['title']
    return hier


def is_annex_article(hierarchy):
    """
    判断当前法条是否属于"附则"章节。
    附则章的法条内容（"本法自X日起施行"等）对检索无价值，整章跳过。
    输入: hierarchy dict
    输出: True 如果该法条位于附则章
    """
    chapter_title = hierarchy.get('章', '')
    return '附则' in chapter_title


# ============================================================
# 法条提取
# ============================================================

def extract_articles(text, law_name, department):
    """
    从法律正文中提取所有法条，产出含完整元数据的 chunk 字典列表。
    输入: text (str) — 预处理后的法律正文
          law_name (str) — 法律全称
          department (str) — 所属法律部门
    输出: [{"chunk_id": ..., "law_name": ..., "article_no": ..., ...}, ...]

    处理流程:
    1. 正则匹配所有"第X条"位置
    2. 两两之间截取 content
    3. 根据法条位置从预扫描的 chapters 分配 hierarchy
    4. 跳过已删除条文、附则条文
    5. 构造 11 字段完整 dict
    6. 去重（同 law_name+article_no 保留 content 最长）
    """
    chapters = extract_chapters(text)
    matches = list(ARTICLE_PAT.finditer(text))
    articles = []

    for i, m in enumerate(matches):
        # 解析条号
        cn_num = m.group(1)
        suffix_cn = m.group(3)
        article_no = cn2arabic(cn_num)
        if suffix_cn:
            article_no += "之" + suffix_cn

        # 截取法条正文
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else min(len(text), start + 2000)
        content = clean_article_content(text[start:end].strip())

        # 跳过无效条文
        if is_deleted_article(content):
            continue

        # 分配层级
        hierarchy = assign_hierarchy(m.start(), chapters)

        # 跳过附则
        if is_annex_article(hierarchy):
            continue

        # 超长内容 → 内嵌的"第X条"重新提取
        if len(content) > 2000:
            inner = _extract_from_segment(content, law_name, department, chapters)
            if inner and len(inner) > 1:
                articles.extend(inner)
                continue

        articles.append({
            "chunk_id": build_chunk_id(law_name, article_no),
            "law_name": law_name,
            "article_no": article_no,
            "article_no_sort_key": parse_sort_key(article_no),
            "content": content,
            "token_count": 0,        # 后填充（需 tokenizer）
            "char_count": len(content),
            "department": department,
            "effective_date": None,  # 后填充（需文件名）
            "hierarchy": hierarchy,
        })

    return _deduplicate_articles(articles)


def _extract_from_segment(text, law_name, department, parent_chapters):
    """
    在过长的文本片段中重新提取内嵌法条（用于修复合并问题）。
    输入/输出同 extract_articles，但 chapters 复用父级扫描结果。
    """
    matches = list(ARTICLE_PAT.finditer(text))
    articles = []
    for i, m in enumerate(matches):
        cn_num = m.group(1)
        suffix_cn = m.group(3)
        article_no = cn2arabic(cn_num)
        if suffix_cn:
            article_no += "之" + suffix_cn

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = clean_article_content(text[start:end].strip())

        if is_deleted_article(content):
            continue

        hierarchy = assign_hierarchy(m.start(), parent_chapters)
        if is_annex_article(hierarchy):
            continue

        articles.append({
            "chunk_id": build_chunk_id(law_name, article_no),
            "law_name": law_name,
            "article_no": article_no,
            "article_no_sort_key": parse_sort_key(article_no),
            "content": content,
            "token_count": 0,
            "char_count": len(content),
            "department": department,
            "effective_date": None,
            "hierarchy": hierarchy,
        })
    return articles


def _deduplicate_articles(articles):
    """
    去重: 同 law_name + article_no 时保留 content 最长的一条。
    输入: articles (list of dict)
    输出: 去重后的 list

    "第X条" 和 "第X条之一" 共用一个主条号时，正则可能在"第X条"
    处截断并产生两段重复片段。较短的一侧通常是误匹配碎片。
    """
    key_to_articles = {}
    for a in articles:
        key = (a['law_name'], a['article_no'])
        key_to_articles.setdefault(key, []).append(a)

    result = []
    for key, dupes in key_to_articles.items():
        if len(dupes) > 1:
            best = max(dupes, key=lambda x: x['char_count'])
            result.append(best)
        else:
            result.append(dupes[0])
    return result


# ============================================================
# 产出验证
# ============================================================

# chunk 必填字段及类型
REQUIRED_FIELDS = [
    'chunk_id', 'law_name', 'article_no', 'article_no_sort_key',
    'content', 'token_count', 'char_count', 'department',
    'effective_date', 'hierarchy',
]

VALID_ARTICLE_NO = re.compile(r'^\d+(之[一二三四五六七八九十]+)?$')
VALID_DATE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def validate_articles(articles):
    """
    对提取产出进行全量字段级校验。
    输入: articles (list of dict)
    输出: (passed_count, failure_descriptions)

    校验规则:
    1. 11 个必填字段存在且非空
    2. chunk_id 全局唯一
    3. article_no 格式: 数字 + 可选的"之X"后缀
    4. effective_date 格式: YYYY-MM-DD
    5. department 在 8 个合法值内
    6. char_count == len(content)
    7. hierarchy 的键仅限于 编/分编/章/节
    8. article_no_sort_key 是 [int, ...]
    9. token_count >= 0
    """
    passed = 0
    failures = []
    seen_chunk_ids = set()
    valid_hierarchy_keys = {'编', '分编', '章', '节'}

    for i, a in enumerate(articles):
        aid = a.get('chunk_id', f'[index={i}]')
        ok = True

        # 1. 必填字段
        for f in REQUIRED_FIELDS:
            if f not in a or a[f] is None:
                failures.append(f"[{aid}] 缺少必填字段: {f}")
                ok = False

        if not ok:
            continue

        # 2. chunk_id 唯一
        cid = a['chunk_id']
        if cid in seen_chunk_ids:
            failures.append(f"[{aid}] chunk_id 重复: {cid}")
            ok = False
        seen_chunk_ids.add(cid)

        # 3. article_no 格式
        if not VALID_ARTICLE_NO.match(a['article_no']):
            failures.append(f"[{aid}] article_no 格式无效: {a['article_no']}")
            ok = False

        # 4. effective_date 格式
        if not VALID_DATE.match(a['effective_date']):
            failures.append(f"[{aid}] effective_date 格式无效: {a['effective_date']}")
            ok = False

        # 5. department 枚举
        if a['department'] not in VALID_DEPARTMENTS:
            failures.append(f"[{aid}] department 非法: {a['department']}")
            ok = False

        # 6. char_count 一致性
        expected_chars = len(a['content'])
        if a['char_count'] != expected_chars:
            failures.append(
                f"[{aid}] char_count 不一致: 声明={a['char_count']}, 实际={expected_chars}")
            ok = False

        # 7. hierarchy 键
        for k in a['hierarchy']:
            if k not in valid_hierarchy_keys:
                failures.append(f"[{aid}] hierarchy 非法键: {k}")
                ok = False

        # 8. sort_key 类型
        sk = a['article_no_sort_key']
        if not isinstance(sk, list) or not all(isinstance(x, int) for x in sk):
            failures.append(f"[{aid}] sort_key 类型错误: {type(sk).__name__}")
            ok = False

        # 9. token_count 界限
        if a['token_count'] < 0:
            failures.append(f"[{aid}] token_count 为负: {a['token_count']}")
            ok = False

        if ok:
            passed += 1

    return passed, failures


# ============================================================
# 主流程
# ============================================================

def extract_struct_summary(text):
    """返回文档结构摘要, e.g. '编→章→节→条'"""
    parts = []
    if re.search(r'第[一二三四五六七八九十百千零]+编', text):
        parts.append('编')
    if re.search(r'第[一二三四五六七八九十百千零]+分编', text):
        parts.append('分编')
    if re.search(r'第[一二三四五六七八九十百千零]+章', text):
        parts.append('章')
    if re.search(r'第[一二三四五六七八九十百千零]+节', text):
        parts.append('节')
    parts.append('条')
    return '→'.join(parts)


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    total_files = 0
    total_articles = 0
    total_filtered = 0      # 附则过滤数
    failed = []
    stats = {}

    with open(OUTPUT, 'w', encoding='utf-8') as out:
        for dept in sorted(os.listdir(LAW_DIR)):
            dept_path = os.path.join(LAW_DIR, dept)
            if not os.path.isdir(dept_path):
                continue

            dept_files = 0
            dept_articles = 0

            for fname in sorted(os.listdir(dept_path)):
                if not fname.endswith('.docx'):
                    continue

                fpath = os.path.join(dept_path, fname)
                law_name = fname.rsplit('_', 1)[0]
                effective_date = extract_effective_date(fname)

                try:
                    raw_text = docx2txt_process(fpath)
                    text = pre_clean_text(raw_text)
                    articles_raw = extract_articles(text, law_name, dept)

                    # 补充 effective_date
                    for a in articles_raw:
                        a['effective_date'] = effective_date

                    # 校验
                    passed, failures_list = validate_articles(articles_raw)
                    if failures_list:
                        for f in failures_list:
                            print(f"  [验证] {f}")

                    for art in articles_raw:
                        out.write(json.dumps(art, ensure_ascii=False) + '\n')

                    dept_files += 1
                    dept_articles += len(articles_raw)

                    struct = extract_struct_summary(text)
                    chars = len(text)
                    print(f"  [{dept}] {law_name[:30]:30s} {len(articles_raw):4d}条  {chars:6d}字  {struct}")

                except Exception as e:
                    failed.append((dept, fname, str(e)))
                    print(f"  [{dept}] {fname}  FAILED: {e}")

            stats[dept] = {"files": dept_files, "articles": dept_articles}
            total_files += dept_files
            total_articles += dept_articles

    # Summary
    print(f"\n{'='*60}")
    print(f"总文件数: {total_files}")
    print(f"总法条数: {total_articles}")
    print(f"成功率: {total_files}/{total_files + len(failed)}")
    if failed:
        print(f"\n失败文件 ({len(failed)}):")
        for dept, fname, err in failed:
            print(f"  [{dept}] {fname}: {err}")

    print(f"\n部门统计:")
    for dept, s in sorted(stats.items()):
        print(f"  {dept:16s}: {s['files']:3d} 部法律, {s['articles']:5d} 条")

    print(f"\n输出: {OUTPUT}")

    # Write stats
    stats_path = OUTPUT.replace('.jsonl', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total_files": total_files,
            "total_articles": total_articles,
            "failed": [{"dept": d, "file": fn, "error": e} for d, fn, e in failed],
            "departments": stats,
        }, f, ensure_ascii=False, indent=2)
    print(f"统计: {stats_path}")

    # 用 MiniMind tokenizer 回填 token_count
    fill_token_counts(OUTPUT)


# ============================================================
# token_count 后填充
# ============================================================

def fill_token_counts(jsonl_path=OUTPUT):
    """
    读取 article_index.jsonl，用 MiniMind tokenizer 逐条编码，
    回填 token_count 字段，原地覆盖写入。

    此函数放在 main() 流程最后一步，只加载一次 tokenizer。
    """
    TOKENIZER_DIR = os.path.join(BASE, "..", "minimind", "model")
    if not os.path.isdir(TOKENIZER_DIR):
        print("[token_count] tokenizer 目录不存在，跳过填充")
        return

    from transformers import AutoTokenizer
    print(f"[token_count] 加载 MiniMind tokenizer: {TOKENIZER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    print(f"  词表大小: {tokenizer.vocab_size}")

    # 读入全部
    with open(jsonl_path, "r", encoding="utf-8") as f:
        articles = [json.loads(line) for line in f if line.strip()]

    print(f"  编码 {len(articles)} 条法条...")
    updated = 0
    for a in articles:
        a["token_count"] = len(tokenizer.encode(a["content"]))
        updated += 1

    # 覆盖写入
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    # 快速统计
    tokens = [a["token_count"] for a in articles]
    print(f"  完成: {updated} 条, token/条 mean={sum(tokens)/len(tokens):.1f}, "
          f"min={min(tokens)}, max={max(tokens)}")


if __name__ == '__main__':
    main()
