"""原文证据子单元的契约与语义边界测试。"""

from dataclasses import replace

import pytest

from rag.knowledge.evidence_units import (
    EvidenceUnit,
    make_unit_id,
    split_article,
    validate_article_units,
)
from rag.knowledge.repository import LegalArticle


def _article(content, *, chunk_id="示例规定#15"):
    law_name, article_no = chunk_id.rsplit("#", 1)
    return LegalArticle(
        chunk_id=chunk_id,
        law_name=law_name,
        article_no=article_no,
        content=content,
        source_type="department_rule",
    )


def test_short_article_uses_exact_full_text_unit():
    article = _article("申请人应当提交真实、完整的材料。")

    units = split_article(article)

    assert len(units) == 1
    assert units[0].unit_type == "full"
    assert units[0].text == article.content
    assert units[0].unit_id == "示例规定#15::span-v1@000000-000016"


def test_numbered_items_depend_on_lead_and_preserve_offsets():
    content = (
        "执法人员应当完成下列任务：\n"
        "（一）检查现场并制作记录；\n"
        "（二）告知当事人依法享有的权利。\n"
        "现场已经被其他机关封存的，不得重复封存。"
    )
    article = _article(content)

    units = split_article(article)

    assert [unit.unit_type for unit in units] == [
        "lead",
        "list_item",
        "list_item",
        "tail",
    ]
    assert units[1].dependency_unit_ids == (units[0].unit_id,)
    assert units[2].dependency_unit_ids == (units[0].unit_id,)
    assert units[3].dependency_unit_ids == ()
    for unit in units:
        assert unit.text == content[unit.start_char : unit.end_char]


def test_inline_numbered_items_are_split_only_for_ordered_sequence():
    article = _article("应当核验：（一）身份信息；（二）申请材料；（三）授权文件。")

    units = split_article(article)

    assert [unit.unit_type for unit in units] == [
        "lead",
        "list_item",
        "list_item",
        "list_item",
    ]
    assert all(
        unit.dependency_unit_ids == (units[0].unit_id,) for unit in units[1:]
    )


def test_dependent_sentence_stays_with_previous_sentence():
    content = (
        "负责机关应当在收到申请后五日内完成审查，并将结果书面告知申请人。"
        "申请材料不完整或者不符合法定形式的，应当一次告知申请人需要补正的全部内容，"
        "并说明补正材料的形式要求、提交方式和办理期限。"
        "但是，法律另有规定的，依照其规定。"
        "无正当理由逾期不处理或者未履行告知义务的，由上级机关责令限期改正；"
        "造成严重后果的，对负有责任的领导人员和直接责任人员依法给予处分。"
    )
    article = _article(content)

    units = split_article(article)

    assert len(units) == 3
    assert units[1].text.endswith("依照其规定。")
    assert units[1].text.startswith("申请材料不完整或者不符合法定形式的")


def test_paragraph_reference_has_explicit_dependency():
    article = _article(
        "主管部门应当在十日内作出决定。\n"
        "前款规定的期限，自受理之日起计算。"
    )

    units = split_article(article)

    assert len(units) == 2
    assert units[1].dependency_unit_ids == (units[0].unit_id,)


def test_reference_inside_tail_depends_on_list_lead():
    article = _article(
        "有下列情形之一的，依法给予处罚：\n"
        "（一）拒绝检查的；\n"
        "（二）隐瞒事实的。\n"
        "实施前款行为并造成严重后果的，依照前款规定从重处罚。"
    )

    units = split_article(article)

    assert units[-1].unit_type == "tail"
    assert units[-1].dependency_unit_ids == (units[0].unit_id,)


def test_multiline_list_item_keeps_continuation_and_shared_lead():
    content = (
        "设备校验周期如下：\n"
        "（一）甲类设备每六个月校验一次；\n"
        "投入使用后三个月内进行监视性校验。\n"
        "（二）乙类设备每十八个月校验一次；\n"
        "投入使用后九个月内进行监视性校验。\n"
        "校验周期届满前后设置十五天时间窗口。"
    )
    article = _article(content)

    units = split_article(article)

    assert [unit.unit_type for unit in units] == [
        "lead",
        "list_item",
        "list_item",
        "tail",
    ]
    assert "\n投入使用后三个月内" in units[1].text
    assert "\n投入使用后九个月内" in units[2].text
    assert units[1].dependency_unit_ids == (units[0].unit_id,)
    assert units[2].dependency_unit_ids == (units[0].unit_id,)


def test_only_colon_sentence_before_block_list_is_lead():
    article = _article(
        "违法种植的，一律予以铲除。有下列情形之一的，依法处罚：\n"
        "（一）数量较大的；\n"
        "（二）经处理后又种植的。"
    )

    units = split_article(article)

    assert [unit.unit_type for unit in units] == [
        "sentence",
        "lead",
        "list_item",
        "list_item",
    ]
    assert units[2].dependency_unit_ids == (units[1].unit_id,)
    assert units[3].dependency_unit_ids == (units[1].unit_id,)


def test_bare_chinese_numbered_items_are_recognized():
    article = _article("办理要求如下：\n一、核验身份；\n二、审查材料；\n三、作出决定。")

    units = split_article(article)

    assert [unit.unit_type for unit in units] == [
        "lead",
        "list_item",
        "list_item",
        "list_item",
    ]
    assert all(
        unit.dependency_unit_ids == (units[0].unit_id,) for unit in units[1:]
    )


def test_unfinished_lines_merge_and_unnumbered_list_depends_on_lead():
    article = _article("委员会由下列人员组成：\n主任，\n副主任若干人，\n委员若干人。")

    units = split_article(article)

    assert len(units) == 2
    assert units[1].text == "主任，\n副主任若干人，\n委员若干人。"
    assert units[1].unit_type == "list_item"
    assert units[1].dependency_unit_ids == (units[0].unit_id,)


def test_referential_lead_keeps_upstream_and_downstream_dependencies():
    article = _article(
        "主管机关可以延长审查期限。\n"
        "前款规定的重大事项包括：\n"
        "（一）控制权发生变化；\n"
        "（二）主要资产被查封。"
    )

    units = split_article(article)

    assert units[1].unit_type == "lead"
    assert units[1].dependency_unit_ids == (units[0].unit_id,)
    assert units[2].dependency_unit_ids == (units[1].unit_id,)


def test_ambiguous_marker_only_layout_falls_back_to_full_article():
    article = _article("材料包括：\n（一）、\n（二）、\n其他材料。")

    units = split_article(article)

    assert len(units) == 1
    assert units[0].unit_type == "full"
    assert units[0].text == article.content


def test_orphan_list_items_fall_back_to_full_article():
    article = _article("一般要求应当遵守。\n（二）缺少第一项；\n（三）缺少引导句。")

    units = split_article(article)

    assert len(units) == 1
    assert units[0].unit_type == "full"


def test_long_nested_numbered_list_has_two_level_dependencies():
    article = _article(
        "申请人应当提交下列材料：\n"
        "（一）身份证明；\n"
        "（二）专业能力证明："
        "1.学历证书以及教育行政部门出具的认证材料，材料应当真实、完整、有效，"
        "并能够清楚反映学习专业、学习年限、毕业时间和证书编号；"
        "该项材料应当由申请人签字确认。\n"
        "2.连续三年的从业经历证明以及所在单位出具的岗位说明，岗位说明应当加盖公章，"
        "并由单位负责人确认申请人的实际工作内容和任职期限；"
        "3.参加专业能力考试并取得合格成绩的证明，考试成绩应当在规定有效期内，"
        "超过有效期的应当按照规定重新参加相应科目的考试；"
        "4.法律、法规规定的其他能够证明专业能力的材料，提交复印件时应当核验原件，"
        "通过电子方式提交的还应当符合电子签名和电子档案管理要求。"
    )

    units = split_article(article)

    assert [unit.unit_type for unit in units] == [
        "lead",
        "list_item",
        "list_item",
        "list_item",
        "list_item",
        "list_item",
        "list_item",
    ]
    outer_lead = units[0].unit_id
    nested_lead = units[2].unit_id
    assert units[2].dependency_unit_ids == (outer_lead,)
    assert all(
        nested_lead in unit.dependency_unit_ids for unit in units[3:]
    )


def test_plain_heading_before_ordered_list_is_a_lead():
    article = _article("申请材料\n（一）身份证明；\n（二）住所证明。")

    units = split_article(article)

    assert [unit.unit_type for unit in units] == [
        "lead",
        "list_item",
        "list_item",
    ]
    assert units[1].dependency_unit_ids == (units[0].unit_id,)


def test_nested_number_before_account_code_is_not_treated_as_decimal():
    details = "本项目的核算范围、登记要求和年终处理方式应当完整记载。" * 5
    article = _article(
        "会计科目的使用\n"
        f"（一）资产类：1.第101号科目 现金。{details}"
        f"2.102号科目 银行存款。{details}"
        f"3.第103号科目 有价证券。{details}"
    )

    units = split_article(article)

    assert len(units) == 5
    assert units[1].text == "（一）资产类："
    assert units[2].text.startswith("1.第101号科目")
    assert units[3].text.startswith("2.102号科目")
    assert units[4].text.startswith("3.第103号科目")


def test_validation_rejects_tampered_text_and_duplicate_dependency():
    article = _article("第一段内容。\n前款规定同时适用于复审程序。")
    units = split_article(article)
    bad_text = replace(units[0], text="被篡改的文本")
    with pytest.raises(ValueError, match="精确复原"):
        validate_article_units(article, (bad_text, units[1]))

    with pytest.raises(ValueError, match="不能重复"):
        EvidenceUnit(
            unit_id=units[1].unit_id,
            parent_chunk_id=units[1].parent_chunk_id,
            unit_index=units[1].unit_index,
            unit_type=units[1].unit_type,
            text=units[1].text,
            start_char=units[1].start_char,
            end_char=units[1].end_char,
            dependency_unit_ids=(units[0].unit_id, units[0].unit_id),
        )


def test_unit_id_rejects_invalid_version_and_is_deterministic():
    first = make_unit_id("示例规定#1", "v1", 2, 9)
    second = make_unit_id("示例规定#1", "v1", 2, 9)

    assert first == second == "示例规定#1::span-v1@000002-000009"
    with pytest.raises(ValueError, match="splitter_version"):
        make_unit_id("示例规定#1", "V 1", 2, 9)


def test_split_is_deterministic():
    article = _article("办理程序如下：\n（一）登记；\n（二）审查；\n（三）决定。")

    assert split_article(article) == split_article(article)
