import arxiv
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def search_arxiv(query: str, max_results: int = 3) -> List[Dict]:
    """
    Search ArXiv database for academic papers across any domain.
    Returns list of dicts with title, url, published date, content, and source_type.
    """
    results = []
    try:
        client = arxiv.Client()
        # Use query directly or clean search term without forcing cat:quant-ph
        search_query = query if ":" in query else f"all:{query}"
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        for r in client.results(search):
            published_str = r.published.strftime("%Y-%m-%d") if r.published else "Unknown"
            results.append({
                "title": r.title.strip(),
                "url": r.entry_id,
                "published": published_str,
                "content": f"Title: {r.title.strip()}\nAuthors: {', '.join([a.name for a in r.authors[:3]])}\nSummary: {r.summary.strip()}",
                "source_type": "Academic (ArXiv)"
            })
    except Exception as e:
        logger.error(f"Error searching ArXiv for query '{query}': {e}")
    return results

def search_web(query: str, max_results: int = 3) -> List[Dict]:
    """
    Search DuckDuckGo / web engine for industry news, benchmarks, and announcements.
    Returns list of dicts with title, url, published date, content, and source_type.
    """
    results = []
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            ddg_results = list(ddgs.text(query, max_results=max_results))
            for r in ddg_results:
                title = r.get("title", "Untitled").strip()
                url = r.get("href") or r.get("link", "#")
                body = r.get("body") or r.get("snippet", "")
                if title:
                    results.append({
                        "title": title,
                        "url": url,
                        "published": "Recent Industry News",
                        "content": f"Title: {title}\nSnippet: {body}",
                        "source_type": "Industry Web"
                    })
    except Exception as e:
        logger.error(f"Error searching Web for query '{query}': {e}")
    return results
