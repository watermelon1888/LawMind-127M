"""法律问题的确定性前置路由。"""
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from rag.core.contracts import (
    AnswerMode,
    BusinessRoute,
    LegalTaskType,
    RouteDecision,
)
from rag.query.exact_reference import (
    count_article_references,
    contains_article_reference,
)


class QueryRoute(str, Enum):
    """进入后续模块前的唯一处理路径。"""

    EXACT_LOOKUP = "exact_lookup"
    SEMANTIC_SEARCH = "semantic_search"
    CLARIFY = "clarify"
    REFUSE = "refuse"


class QueryReason(str, Enum):
    """策略拒答使用的闭合原因码。"""

    NON_LEGAL = "non_legal"
    TIME_SENSITIVE = "time_sensitive"
    UNSUPPORTED_LEGAL_SOURCE = "unsupported_legal_source"
    UNSUPPORTED_LEGAL_TASK = "unsupported_legal_task"


@dataclass(frozen=True)
class QueryDecision:
    """保留原始问题的不可变路由结果。"""

    query: str
    route: QueryRoute
    reason: Optional[QueryReason] = None

    def __post_init__(self):
        if not isinstance(self.query, str):
            raise TypeError("query 必须是字符串")
        if not isinstance(self.route, QueryRoute):
            raise TypeError("route 必须是 QueryRoute")
        if self.reason is not None and not isinstance(
            self.reason,
            QueryReason,
        ):
            raise TypeError("reason 必须是 QueryReason 或 None")
        if self.route is QueryRoute.REFUSE and self.reason is None:
            raise ValueError("refuse 路由必须携带 reason")
        if self.route is not QueryRoute.REFUSE and self.reason is not None:
            raise ValueError("只有 refuse 路由可以携带 reason")


_HISTORICAL_VERSION_PHRASES = (
    "修法前",
    "修改前",
    "修订前",
    "旧法",
    "旧版",
    "历史版本",
    "废止前",
    "失效前",
    "原来怎么规定",
    "以前怎么规定",
    "当时怎么规定",
    "当时如何规定",
    "那时怎么规定",
    "当年的规定",
)
_HISTORICAL_APPLICABILITY_CUES = (
    "当时",
    "那时",
    "彼时",
    "签订时",
    "订立时",
    "发生时",
    "被罚时",
    "受伤时",
    "适用什么",
    "应适用",
    "应当适用",
    "按当时",
    "按彼时",
)
_PAST_TIME_RE = re.compile(
    r"(?:19|20)\d{2}(?:年(?:\d{1,2}月(?:\d{1,2}日)?)?|[-./]\d{1,2}(?:[-./]\d{1,2})?)"
    r"|(?:\d{1,2}年前|上一年|上年|去年|前年|那年|当年)"
)
_CUTOFF_YEAR_RE = re.compile(r"(?:截至|截止至|截止到)(?:19|20)\d{2}年")
_YEAR_RE = re.compile(r"(?:19|20)\d{2}年")
_BOOK_TITLE_FOR_TIME_RE = re.compile(r"《[^《》]*》")
_ENACTMENT_YEAR_CONTEXT_RE = re.compile(
    r"^(?:19|20)\d{2}年(?:通过|颁布|公布|施行|生效)"
    r"(?:并(?:公布|施行|生效))?的"
)
_HISTORICAL_RULE_OBJECT_RE = re.compile(
    r"(?:《[^《》]+》|[\u4e00-\u9fff]{1,20}(?:法|法典)|法律|法规|法条|条文)"
)
_HISTORICAL_RULE_QUESTION_RE = re.compile(
    r"(?:(?:怎么|如何|有何|有什么|是怎样(?:的)?)规定|规定(?:是什么|有哪些|了什么))"
)

_EXPLICIT_NON_LEGAL_PATTERNS = (
    re.compile(r"(?:今天天气|明天天气|天气怎么样|天气预报)"),
    re.compile(r"(?:菜谱|怎么做菜|做饭教程|家常菜|推荐[^，。！？?]{0,8}菜)"),
    re.compile(r"(?:游戏攻略|怎么通关|游戏推荐)"),
    re.compile(r"(?:星座|算命)"),
    re.compile(r"(?:减肥计划|健身计划|穿搭推荐|化妆教程)"),
    re.compile(r"(?:数学题|解方程|求导|积分题)"),
    re.compile(r"(?:销量|销售额|市场份额|市场规模|保费收入)"),
    re.compile(r"(?:英语|外语|口语).{0,12}(?:怎么学|学好|学习)|怎么.{0,8}学好英语"),
    re.compile(r"(?:Python|Java|编程语言).{0,16}(?:初学者|学习|选择)"),
    re.compile(r"(?:股市|股票).{0,12}(?:涨|跌|行情|走势)"),
    re.compile(r"(?:高铁|火车|飞机).{0,12}(?:多久|多长时间|时长)"),
    re.compile(r"(?:感冒|发烧).{0,12}(?:吃什么药|用什么药|好得快)"),
    re.compile(r"(?:手机|相机).{0,12}(?:拍照|推荐|最好)"),
    re.compile(r"(?:帮我|请|给我)(?:写|创作).{0,12}(?:诗|小说|故事)"),
)
_LEGAL_SIGNALS = (
    "法律",
    "法条",
    "条文",
    "法规",
    "刑法",
    "民法",
    "诉讼",
    "合同",
    "赔偿",
    "补偿",
    "处罚",
    "合法",
    "违法",
    "犯罪",
    "权利",
    "义务",
    "侵权",
    "劳动",
    "婚姻",
    "公司法",
    "股东",
    "董事",
    "清算",
    "税务",
    "仲裁",
    "责任",
    "被辞退",
    "被裁员",
    "被盗",
    "诈骗",
    "拒赔",
    "理赔",
    "保险合同",
    "隐私",
    "版权",
)
_INTENT_SEPARATOR_RE = re.compile(
    r"(?:[，,。；;！？!?]+|再|同时|另外|然后|顺便)"
)
_UNSUPPORTED_SOURCE_RE = re.compile(
    r"(?:行政法规|监察法规|地方性法规|部门规章|司法解释|指导性案例|判例|外国法律|外国法)"
)
_LEGAL_DOCUMENT_PATTERN = (
    r"(?:起诉状|答辩状|上诉状|申诉书|律师函|合同(?!法))"
)
_DOCUMENT_TASK_RE = re.compile(
    rf"(?:帮我|替我|给我|为我|请)(?:代写|撰写|起草|制作)"
    rf"(?:一份|一个|个)?[^，。！？?]{{0,12}}{_LEGAL_DOCUMENT_PATTERN}"
    rf"|(?:帮我|替我|给我|为我|请)写(?!(?:一下)?合同法)"
    rf"(?:一份|一个|个)?"
    rf"[^，。！？?]{{0,12}}{_LEGAL_DOCUMENT_PATTERN}"
    rf"|(?:帮我|替我|给我|为我|请)(?:完整)?(?:审阅|审核|修改)"
    rf"(?:这份|一份|整份|完整)?[^，。！？?]{{0,8}}{_LEGAL_DOCUMENT_PATTERN}"
)
_OUTCOME_PREDICTION_RE = re.compile(
    r"(?:预测|估算|预判)[^，。！？?]{0,12}"
    r"(?:胜诉率|胜诉概率|败诉概率|会不会胜诉|能否胜诉|会判几年|判刑多久)"
    r"|(?:胜诉率|胜诉概率)[^，。！？?]{0,8}(?:多少|多高)"
    r"|(?:保证|确保)[^，。！？?]{0,12}(?:一定|肯定)?胜诉"
)
_REGULATORY_EVASION_RE = re.compile(
    r"(?:规避|绕过|逃避)[^，。！？?]{0,16}(?:监管|检查|审查|执法|处罚)"
)
_BROAD_SCOPE_PATTERNS = (
    re.compile(
        r"(?:全部|所有|全面)[^，。！？?]{0,12}"
        r"(?:规定|制度|内容)[^，。！？?]{0,12}(?:讲|介绍|解释)"
    ),
    re.compile(r"(?:法|法典)[^，。！？?]{0,8}(?:都有哪些规定|所有规定|全部规定)"),
    re.compile(r"(?:所有权利|全部权利).{0,20}(?:完整讲|全部讲|讲清楚)"),
    re.compile(r"(?:法律要求|法律规定).{0,12}(?:全部|所有)(?:列出|列出来|讲清楚)"),
)
_MISSING_REFERENT_RE = re.compile(
    r"^(?:请问)?(?:(?:他|她|对方)?(?:这样|这么做|这种情况|这件事)"
    r"(?:做)?(?:算|是|属于|构成|是否|违法|合法|犯罪|应该怎么办|怎么办|该怎么办)"
    r"|(?:这个|那个)(?:能|可以|是否|算|属于|构成|应该|该)?"
    r"(?:要求赔偿|起诉|怎么办|违法|合法|犯罪))"
)
_UNSPECIFIED_PAYMENT_RE = re.compile(
    r"^(?:老板|房东|对方|公司|单位)[^，。！？?]{0,6}"
    r"(?:不给|不退|扣着|扣了)(?:钱|款)(?:怎么办|该怎么办)?[？?]?$"
)
_INCOMPLETE_TOPIC_RE = re.compile(
    r"^(?:上一年|上年|去年|前年|那年|当年)的?(?:保险|合同|工资|房租|押金)[。！？?]?$"
)
_INCOMPLETE_CASE_RE = re.compile(
    r"(?:涉嫌犯罪|犯了罪|犯罪了)[^。！？?]{0,12}"
    r"(?:最高|最多)?(?:可能)?判(?:多少年|多久)"
)
_UNAVAILABLE_DOCUMENT_RE = re.compile(
    r"(?:我签的)?这份合同[^，。！？?]{0,12}(?:有没有法律效力|是否有效|有效吗)"
)
_IMPRECISE_ARTICLE_RE = re.compile(
    r"第[零〇一二三四五六七八九十百千万两\d]*几条|记不清[^，。！？?]{0,8}条号"
)
_CONTENT_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
MAX_QUERY_CHARS = 256


def _has_year_qualified_article_reference(compact_query):
    query_without_law_titles = _BOOK_TITLE_FOR_TIME_RE.sub(
        "《法名》",
        compact_query,
    )
    for year in _YEAR_RE.finditer(query_without_law_titles):
        tail = query_without_law_titles[year.start() :]
        if _ENACTMENT_YEAR_CONTEXT_RE.match(tail):
            continue
        if contains_article_reference(tail[:80]):
            return True
    return False


def _has_past_rule_query(compact_query):
    for past_time in _PAST_TIME_RE.finditer(compact_query):
        time_scope = compact_query[past_time.start() :]
        if _ENACTMENT_YEAR_CONTEXT_RE.match(time_scope):
            continue
        tail = compact_query[past_time.end() : past_time.end() + 80]
        if _HISTORICAL_RULE_OBJECT_RE.search(
            tail
        ) and _HISTORICAL_RULE_QUESTION_RE.search(tail):
            return True
    return False


def _depends_on_historical_law(compact_query):
    if any(phrase in compact_query for phrase in _HISTORICAL_VERSION_PHRASES):
        return True
    if _CUTOFF_YEAR_RE.search(compact_query):
        return True
    if _has_year_qualified_article_reference(compact_query):
        return True
    if _has_past_rule_query(compact_query):
        return True
    return bool(
        _PAST_TIME_RE.search(compact_query)
        and any(cue in compact_query for cue in _HISTORICAL_APPLICABILITY_CUES)
    )


def _has_legal_signal(compact_query):
    return contains_article_reference(compact_query) or any(
        signal in compact_query for signal in _LEGAL_SIGNALS
    )


def _has_non_legal_signal(compact_query):
    return any(
        pattern.search(compact_query) for pattern in _EXPLICIT_NON_LEGAL_PATTERNS
    )


def _is_unsupported_task(compact_query):
    return any(
        pattern.search(compact_query)
        for pattern in (
            _DOCUMENT_TASK_RE,
            _OUTCOME_PREDICTION_RE,
            _REGULATORY_EVASION_RE,
        )
    )


def _has_mixed_intents(compact_query):
    clauses = tuple(
        clause
        for clause in _INTENT_SEPARATOR_RE.split(compact_query)
        if clause
    )
    if len(clauses) < 2:
        return False
    has_legal_clause = any(_has_legal_signal(clause) for clause in clauses)
    has_other_clause = any(
        _is_unsupported_task(clause)
        or (_has_non_legal_signal(clause) and not _has_legal_signal(clause))
        for clause in clauses
    )
    return has_legal_clause and has_other_clause


def _requires_clarification(compact_query):
    return _clarification_reason(compact_query) is not None


def _clarification_reason(compact_query):
    if any(pattern.search(compact_query) for pattern in _BROAD_SCOPE_PATTERNS):
        return "scope_too_broad"
    if _MISSING_REFERENT_RE.search(compact_query):
        return "missing_referent"
    if _UNSPECIFIED_PAYMENT_RE.search(compact_query):
        return "missing_key_facts"
    if _INCOMPLETE_TOPIC_RE.search(compact_query):
        return "incomplete_topic"
    if _INCOMPLETE_CASE_RE.search(compact_query):
        return "missing_key_facts"
    if _UNAVAILABLE_DOCUMENT_RE.search(compact_query):
        return "unavailable_document"
    if _IMPRECISE_ARTICLE_RE.search(compact_query):
        return "ambiguous_article_reference"
    return None


def route_query(query):
    """按固定优先级把原始问题路由到唯一处理路径。"""
    if not isinstance(query, str):
        raise TypeError("query 必须是字符串")
    if not _CONTENT_RE.search(query):
        return RouteDecision(query, BusinessRoute.CLARIFY, reason="empty_input")

    compact_query = re.sub(r"\s+", "", query)
    if len(compact_query) > MAX_QUERY_CHARS:
        return RouteDecision(
            query,
            BusinessRoute.CLARIFY,
            reason="query_too_long",
        )
    has_legal_signal = _has_legal_signal(compact_query)
    has_non_legal_signal = _has_non_legal_signal(compact_query)
    has_unsupported_task = _is_unsupported_task(compact_query)

    if _has_mixed_intents(compact_query):
        return RouteDecision(
            query,
            BusinessRoute.CLARIFY,
            reason="mixed_intent",
        )
    if has_unsupported_task:
        return RouteDecision(
            query,
            BusinessRoute.CLARIFY,
            reason=QueryReason.UNSUPPORTED_LEGAL_TASK.value,
        )
    if _UNSUPPORTED_SOURCE_RE.search(compact_query):
        return RouteDecision(
            query,
            BusinessRoute.CLARIFY,
            reason=QueryReason.UNSUPPORTED_LEGAL_SOURCE.value,
        )
    if _depends_on_historical_law(compact_query) and not (
        has_non_legal_signal and not has_legal_signal
    ):
        return RouteDecision(
            query,
            BusinessRoute.CLARIFY,
            reason=QueryReason.TIME_SENSITIVE.value,
        )
    if has_non_legal_signal and not has_legal_signal:
        return RouteDecision(
            query,
            BusinessRoute.GENERAL_CHAT,
            reason=QueryReason.NON_LEGAL.value,
        )
    if _requires_clarification(compact_query):
        return RouteDecision(
            query,
            BusinessRoute.CLARIFY,
            reason=_clarification_reason(compact_query),
        )
    if contains_article_reference(compact_query):
        if count_article_references(compact_query) > 3:
            return RouteDecision(
                query,
                BusinessRoute.CLARIFY,
                reason="scope_too_broad",
            )
        return RouteDecision(
            query,
            BusinessRoute.ANSWER,
            AnswerMode.EXACT_LOOKUP,
        )
    if not has_legal_signal:
        return RouteDecision(
            query,
            BusinessRoute.CLARIFY,
            reason="external_analysis_required",
        )
    return RouteDecision(
        query,
        BusinessRoute.ANSWER,
        AnswerMode.RETRIEVAL,
    )


__all__ = [
    "AnswerMode",
    "BusinessRoute",
    "LegalTaskType",
    "QueryDecision",
    "QueryReason",
    "QueryRoute",
    "RouteDecision",
    "MAX_QUERY_CHARS",
    "route_query",
]
