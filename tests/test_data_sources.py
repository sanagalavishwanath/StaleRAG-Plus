import datetime as dt
import unittest

from stalerag.data.builder import attach_distractors, build_benchmark
from stalerag.data.revisions import (
    WikipediaClient,
    find_evidence_sentence,
    strip_wikitext,
)
from stalerag.data.wikidata import (
    FetchError,
    Statement,
    WikidataClient,
    group_conflicts,
)
from tests.helpers import FakeResponse, FakeSession


def binding(item, holder, start=None, end=None, qid=False):
    row = {
        "itemLabel": {"value": "Q123" if qid else item},
        "article": {"value": f"https://en.wikipedia.org/wiki/{item.replace(' ', '_')}"},
        "holderLabel": {"value": holder},
    }
    if start:
        row["start"] = {"value": f"{start}T00:00:00Z"}
    if end:
        row["end"] = {"value": f"{end}T00:00:00Z"}
    return row


def sparql_payload(*bindings):
    return {"results": {"bindings": list(bindings)}}


class WikitextTests(unittest.TestCase):
    def test_strip_wikitext(self):
        raw = (
            "'''[[Acme Corp|Acme]]''' is {{nowrap|a {{small|firm}}}} "
            "in [[Ohio]].<ref name=a>cite</ref>"
            "<ref name=b/> <!-- hidden --> See [https://x.org the site]. [[File:Logo.png|thumb]]"
        )
        cleaned = strip_wikitext(raw)
        self.assertIn("Acme is in Ohio.", cleaned)
        self.assertIn("See the site.", cleaned)
        for fragment in ("{{", "<ref", "hidden", "File:", "'''"):
            self.assertNotIn(fragment, cleaned)

    def test_find_evidence_sentence(self):
        text = "Acme Corp is a company. Jane Doe is the chief executive officer of Acme Corp. More."
        found = find_evidence_sentence(text, ("Jane Doe", "Acme Corp"))
        self.assertEqual(found, "Jane Doe is the chief executive officer of Acme Corp.")
        self.assertIsNone(find_evidence_sentence(text, ("John Roe", "Acme Corp")))
        self.assertIsNone(find_evidence_sentence("Jane Doe, Acme Corp.", ("Jane Doe",), min_len=30))
        self.assertIsNone(find_evidence_sentence("x" * 500 + " Jane Doe.", ("Jane Doe",)))


class WikidataTests(unittest.TestCase):
    def test_fetch_parses_and_skips_unlabeled(self):
        payload = sparql_payload(
            binding("Acme Corp", "Jane Doe", start="2020-01-01"),
            binding("Acme Corp", "Old Boss", start="2010-01-01", end="2019-12-31"),
            binding("Bad Item", "Someone", qid=True),
        )
        session = FakeSession(lambda url, params: FakeResponse(payload))
        statements = WikidataClient(session, pause=0).fetch_statements("P169")
        self.assertEqual(len(statements), 2)
        self.assertEqual(statements[0].title, "Acme Corp")
        self.assertEqual(statements[1].end, dt.date(2019, 12, 31))
        self.assertIn("P169", session.calls[0]["params"]["query"])

    def test_http_error_raises_fetch_error(self):
        session = FakeSession(lambda url, params: FakeResponse({}, status_code=429))
        with self.assertRaises(FetchError):
            WikidataClient(session, pause=0).fetch_statements("P6")

    def test_malformed_date_becomes_none(self):
        payload = sparql_payload(binding("Acme Corp", "Jane Doe", start="2020-01-01"))
        payload["results"]["bindings"][0]["start"]["value"] = "garbage"
        session = FakeSession(lambda url, params: FakeResponse(payload))
        statements = WikidataClient(session, pause=0).fetch_statements("P6")
        self.assertIsNone(statements[0].start)

    def test_group_conflicts(self):
        today = dt.date(2026, 6, 1)
        stmts = [
            Statement("A", "A", "New Person", dt.date(2020, 1, 1), None),
            Statement("A", "A", "Old Person", dt.date(2010, 1, 1), dt.date(2019, 12, 31)),
            Statement("B", "B", "Same Person", dt.date(2020, 1, 1), None),
            Statement("B", "B", "Same Person", dt.date(2010, 1, 1), dt.date(2019, 12, 31)),
            Statement("C", "C", "Only Current", dt.date(2020, 1, 1), None),
            Statement("D", "D", "Undated Current", None, None),
            Statement("D", "D", "Past", dt.date(2000, 1, 1), dt.date(2005, 1, 1)),
            Statement("E", "E", "Future End", dt.date(2020, 1, 1), None),
            Statement("E", "E", "Later", dt.date(2021, 1, 1), dt.date(2030, 1, 1)),
        ]
        records = group_conflicts(stmts, "mayor", today=today)
        self.assertEqual([r.title for r in records], ["A"])
        self.assertEqual(records[0].stale_holder, "Old Person")
        self.assertEqual(records[0].change_date, dt.date(2019, 12, 31))


OLD_PAGE = (
    "'''Acme Corp''' is a firm.<ref>x</ref> "
    "Old Boss is the chief executive officer of Acme Corp."
)
NEW_PAGE = (
    "'''Acme Corp''' is a firm. "
    "Jane Doe is the chief executive officer of Acme Corp, succeeding Old Boss."
)


def wikipedia_handler(missing=()):
    def handler(url, params):
        if "wikidata" in url:
            return FakeResponse(
                sparql_payload(
                    binding("Acme Corp", "Jane Doe", start="2020-01-01"),
                    binding("Acme Corp", "Old Boss", start="2010-01-01", end="2019-12-31"),
                    binding("Zeta Inc", "Zed Quinn", start="2021-01-01"),
                    binding("Zeta Inc", "Ann Past", start="2011-01-01", end="2020-12-31"),
                    binding("Gone Ltd", "New Gone", start="2021-01-01"),
                    binding("Gone Ltd", "Old Gone", start="2011-01-01", end="2020-12-31"),
                )
            )
        title = params["titles"]
        if title in missing:
            return FakeResponse({"query": {"pages": [{"title": title, "missing": True}]}})
        is_old = "rvstart" in params
        if title == "Zeta Inc":
            content = "Zeta Inc makes widgets for everyone around."
        else:
            content = OLD_PAGE if is_old else NEW_PAGE
        stamp = "2019-11-01T00:00:00Z" if is_old else "2026-05-01T00:00:00Z"
        revision = {"timestamp": stamp, "slots": {"main": {"content": content}}}
        page = {"title": title, "revisions": [revision]}
        return FakeResponse({"query": {"pages": [page]}})

    return handler


class BuilderTests(unittest.TestCase):
    def build(self, missing=(), target=10):
        session = FakeSession(wikipedia_handler(missing))
        return build_benchmark(
            WikidataClient(session, pause=0),
            WikipediaClient(session, pause=0),
            target_size=target,
            properties={"P169": "chief executive officer"},
        )

    def test_builds_conflict_pairs_and_skips_unusable_pages(self):
        examples = self.build(missing=("Gone Ltd",))
        self.assertEqual([e.entity for e in examples], ["Acme Corp"])
        example = examples[0]
        self.assertEqual(example.current_answer, "Jane Doe")
        self.assertEqual(example.stale_answer, "Old Boss")
        self.assertEqual(example.passage("stale").date, "2019-11-01")
        self.assertEqual(example.passage("current").date, "2026-05-01")
        self.assertNotIn("Jane Doe", example.passage("stale").text)

    def test_stale_revision_requested_before_the_change(self):
        session = FakeSession(wikipedia_handler())
        WikipediaClient(session, pause=0).revision("Acme Corp", dt.date(2019, 12, 1))
        params = session.calls[0]["params"]
        self.assertEqual(params["rvstart"], "2019-12-01T00:00:00Z")
        self.assertEqual(params["rvdir"], "older")

    def test_missing_page_returns_none_and_http_error_raises(self):
        session = FakeSession(wikipedia_handler(missing=("Acme Corp",)))
        self.assertIsNone(WikipediaClient(session, pause=0).revision("Acme Corp"))
        bad = FakeSession(lambda url, params: FakeResponse({}, status_code=500))
        with self.assertRaises(FetchError):
            WikipediaClient(bad, pause=0).revision("Acme Corp")

    def test_target_size_is_respected(self):
        self.assertEqual(self.build(target=0), [])

    def test_attach_distractors_uses_another_examples_passage(self):
        session = FakeSession(wikipedia_handler())
        examples = build_benchmark(
            WikidataClient(session, pause=0),
            WikipediaClient(session, pause=0),
            target_size=5,
            properties={"P169": "chief executive officer"},
        )
        self.assertEqual(attach_distractors(examples), examples)
        a, b = examples[0], examples[0]
        self.assertLessEqual(len(a.distractors()), 0 if len(examples) < 2 else 1)
        self.assertIs(a, b)


if __name__ == "__main__":
    unittest.main()
