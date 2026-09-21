"""精确法条引用提取的纯单元测试。"""
import unittest

from rag.query.exact_reference import (
    ExactReferenceStatus,
    count_article_references,
    extract_exact_references,
)


class TestExactReferenceExtraction(unittest.TestCase):
    def assert_references(self, query, expected):
        result = extract_exact_references(query)
        self.assertIs(ExactReferenceStatus.FOUND, result.status)
        self.assertEqual(expected, tuple(
            (item.law_name, item.article_no) for item in result.references
        ))

    def test_extracts_single_quoted_or_bare_reference(self):
        cases = (
            (
                "请告诉我《中华人民共和国消防法》第 60 条的条文内容",
                (("中华人民共和国消防法", "第60条"),),
            ),
            (
                "刑法第一百三十三条之一怎么规定？",
                (("刑法", "第一百三十三条之一"),),
            ),
            (
                "2020年通过的民法典第577条是什么？",
                (("民法典", "第577条"),),
            ),
        )
        for query, expected in cases:
            with self.subTest(query=query):
                self.assert_references(query, expected)

    def test_extracts_repeated_and_shared_article_numbers(self):
        cases = (
            "《中华人民共和国刑法》第264条和第266条是什么？",
            "《中华人民共和国刑法》第264、266条是什么？",
            "《中华人民共和国刑法》第264条、266条是什么？",
        )
        expected = (
            ("中华人民共和国刑法", "第264条"),
            ("中华人民共和国刑法", "第266条"),
        )
        for query in cases:
            with self.subTest(query=query):
                self.assert_references(query, expected)

    def test_extracts_up_to_three_references_in_user_order(self):
        self.assert_references(
            "请对比民法典第577、584、585条。",
            (
                ("民法典", "第577条"),
                ("民法典", "第584条"),
                ("民法典", "第585条"),
            ),
        )

    def test_extracts_references_from_different_laws(self):
        self.assert_references(
            "比较《中华人民共和国刑法》第264条和《中华人民共和国民法典》第577条。",
            (
                ("中华人民共和国刑法", "第264条"),
                ("中华人民共和国民法典", "第577条"),
            ),
        )

    def test_incomplete_or_ambiguous_reference_returns_no_partial_result(self):
        queries = (
            "第264条是什么？",
            "刑法和民法典第264条是什么？",
            "《中华人民共和国刑法》和《中华人民共和国民法典》第264条是什么？",
            "刑法第1、2、3、4条是什么？",
        )
        for query in queries:
            with self.subTest(query=query):
                result = extract_exact_references(query)
                self.assertIs(
                    ExactReferenceStatus.CLARIFICATION_REQUIRED,
                    result.status,
                )
                self.assertEqual((), result.references)

    def test_reference_count_understands_coordinated_forms(self):
        cases = (
            ("刑法第264条", 1),
            ("刑法第264、266条", 2),
            ("刑法第264条、266条、第267条", 3),
            ("刑法第1、2、3、4条", 4),
        )
        for query, expected in cases:
            with self.subTest(query=query):
                self.assertEqual(expected, count_article_references(query))


if __name__ == "__main__":
    unittest.main(verbosity=2)
