"""Test per EdgarEftsSearcher — parsing risposta EFTS e costruzione URL."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from requests import HTTPError

from src.edgar.search import EdgarEftsSearcher


class TestBuildUrl:
    def test_build_url_with_cik_and_docfilename(self):
        hit = {
            "_id": "0001213900-26-003766:ea0272591-01_424b2.htm",
            "_source": {
                "adsh": "0001213900-26-003766",
                "ciks": ["1884152"],
                "display_names": ["GS Finance Corp (CIK 0001884152)"],
                "file_date": "2026-01-15",
                "form": "424B2",
            },
        }
        result = EdgarEftsSearcher._build_url(hit)
        assert result["accession_no"] == "0001213900-26-003766"
        assert result["entity_id"] == "1884152"
        assert result["entity_name"] == "GS Finance Corp"
        assert result["filing_date"] == "2026-01-15"
        assert "000121390026003766" in result["url"]
        assert "ea0272591-01_424b2.htm" in result["url"]

    def test_build_url_no_docfilename_fallback_to_index(self):
        hit = {
            "_id": "0001213900-26-003766",
            "_source": {
                "adsh": "0001213900-26-003766",
                "ciks": ["1884152"],
                "display_names": ["Test Entity"],
                "file_date": "2026-02-01",
            },
        }
        result = EdgarEftsSearcher._build_url(hit)
        assert "index.htm" in result["url"]
        assert result["entity_name"] == "Test Entity"

    def test_build_url_no_ciks(self):
        hit = {
            "_id": "abc:doc.htm",
            "_source": {
                "adsh": "abc",
                "display_names": [],
                "file_date": "2026-03-01",
            },
        }
        result = EdgarEftsSearcher._build_url(hit)
        assert result["entity_id"] == ""
        assert result["url"]  # non crasha

    def test_build_url_cik_non_numeric(self):
        hit = {
            "_id": "abc:doc.htm",
            "_source": {
                "adsh": "abc",
                "ciks": ["NOT_A_NUMBER"],
                "display_names": ["Bad CIK Entity"],
                "file_date": "2026-04-01",
            },
        }
        result = EdgarEftsSearcher._build_url(hit)
        assert result["entity_id"] == "NOT_A_NUMBER"
        assert "0/" in result["url"]


class TestDedup:
    def test_collect_all_filings_dedup_by_accession_no(self):
        searcher = EdgarEftsSearcher()
        filing_a = {
            "accession_no": "0001",
            "url": "https://sec.gov/doc_a.htm",
            "entity_name": "A",
        }
        filing_b = {
            "accession_no": "0001",
            "url": "https://sec.gov/doc_a_alt.htm",
            "entity_name": "A",
        }
        filing_c = {
            "accession_no": "0002",
            "url": "https://sec.gov/doc_c.htm",
            "entity_name": "C",
        }
        with patch.object(searcher, "search", return_value=[filing_a, filing_b, filing_c]):
            result = searcher.collect_all_filings()
            assert len(result) == 2
            accessions = {f["accession_no"] for f in result}
            assert accessions == {"0001", "0002"}

    def test_collect_all_filings_empty(self):
        searcher = EdgarEftsSearcher()
        with patch.object(searcher, "search", return_value=[]):
            result = searcher.collect_all_filings()
            assert result == []


class TestSearchPagination:
    def test_search_single_page(self):
        searcher = EdgarEftsSearcher()
        mock_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "a:b.htm",
                        "_source": {
                            "adsh": "a",
                            "ciks": ["123"],
                            "display_names": ["Entity X"],
                            "file_date": "2026-01-01",
                            "form": "424B2",
                        },
                    }
                ],
            }
        }
        with patch.object(searcher, "_get", return_value=mock_response):
            results = searcher.search("IBIT")
            assert len(results) == 1
            assert results[0]["entity_name"] == "Entity X"

    def test_search_empty_results(self):
        searcher = EdgarEftsSearcher()
        mock_response = {
            "hits": {
                "total": {"value": 0},
                "hits": [],
            }
        }
        with patch.object(searcher, "_get", return_value=mock_response):
            results = searcher.search("IBIT")
            assert results == []

    def test_search_pagination_two_pages(self):
        searcher = EdgarEftsSearcher()
        page1 = {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {
                        "_id": "a1:b1.htm",
                        "_source": {
                            "adsh": "a1", "ciks": ["1"], "display_names": ["E1"],
                            "file_date": "2026-01-01", "form": "424B2",
                        },
                    }
                ],
            }
        }
        page2 = {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {
                        "_id": "a2:b2.htm",
                        "_source": {
                            "adsh": "a2", "ciks": ["2"], "display_names": ["E2"],
                            "file_date": "2026-01-02", "form": "424B2",
                        },
                    }
                ],
            }
        }
        with patch.object(searcher, "_get", side_effect=[page1, page2]):
            results = searcher.search("IBIT")
            assert len(results) == 2


class TestSearchError:
    def test_search_http_error_propagates(self):
        searcher = EdgarEftsSearcher()
        with patch.object(searcher, "_get", side_effect=HTTPError("503 Server Error")):
            with pytest.raises(HTTPError):
                searcher.search("IBIT")
