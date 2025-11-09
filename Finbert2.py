"""
yahoo_crypto_sentiment.py
=========================

Fetches broad Yahoo Finance RSS feeds, filters for cryptocurrency-related
articles, runs FinBERT sentiment analysis, and saves results to CSV.

Requirements:
    pip install transformers feedparser requests beautifulsoup4

Usage:
    python yahoo_crypto_sentiment.py
"""

import re
import time
import csv
import requests
import feedparser
from bs4 import BeautifulSoup
from transformers import pipeline


# -------------------- CONFIG --------------------

CRYPTO_KEYWORDS = [
    # General crypto terms
    "crypto", "cryptocurrency", "bitcoin", "btc", "ethereum", "eth",
    "web3", "blockchain", "defi", "stablecoin", "staking", "mining",
    "hashrate", "airdrops", "token", "nft",
    # Entities and tickers
    "binance", "coinbase", "tether", "usdt", "usdc", "solana", "sol",
    "xrp", "avax", "ada", "doge", "etf", "spot bitcoin etf",
    "sec", "cftc", "genesis", "grayscale"
]

# Wide selection of tickers (crypto + tech + related equities)
UNIVERSE = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD",
    "AVAX-USD", "BNB-USD", "COIN", "MSTR", "RIOT", "MARA", "NVDA",
    "AAPL", "MSFT", "AMZN", "META"
]

# Yahoo Finance RSS feeds (site-wide + multi-ticker)
RSS_FEEDS = [
    "https://finance.yahoo.com/news/rssindex",
    f"https://finance.yahoo.com/rss/headline?s={','.join(UNIVERSE)}"
]

# Output file
CSV_FILE = "yahoo_crypto_sentiment.csv"

# User-Agent header (helps prevent request blocks)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-GB,en;q=0.9",
}

# -------------------- HELPERS --------------------

def strip_html(s: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    return re.sub(r"<[^>]+>", " ", s or "").replace("&nbsp;", " ").strip()


def is_crypto_related(text: str) -> bool:
    """Check if text contains any crypto keyword."""
    t = text.lower()
    return any(k in t for k in CRYPTO_KEYWORDS)


def fetch(url: str, timeout: int = 15) -> str:
    """GET request with headers and return text."""
    r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.text


def extract_main_text_from_yahoo(html: str) -> str:
    """Extract article body text from Yahoo Finance page HTML."""
    soup = BeautifulSoup(html, "html.parser")
    selectors = [
        'div.caas-body p',
        'div[data-test="article-content"] p',
        'article p',
    ]
    for sel in selectors:
        ps = soup.select(sel)
        if ps:
            text = " ".join(p.get_text(" ", strip=True) for p in ps)
            if len(text.split()) > 80:
                return re.sub(r"\s+", " ", text).strip()
    # fallback: all <p>
    text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
    return re.sub(r"\s+", " ", text).strip()


# -------------------- MAIN PIPELINE --------------------

def main():
    print("\n--- Yahoo Finance Crypto Sentiment ---\n")

    # Load FinBERT model
    print("Loading FinBERT model...")
    finbert = pipeline("text-classification", 
                           model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
    truncation=True,
    max_length=512
)

    # Parse feeds
    entries = []
    seen_links = set()
    print("Fetching RSS feeds...")

    for rss in RSS_FEEDS:
        feed = feedparser.parse(rss)
        for e in feed.entries:
            link = e.get("link", "")
            if not link or link in seen_links:
                continue
            seen_links.add(link)
            title = e.get("title", "")
            summary = strip_html(e.get("summary", ""))
            published = e.get("published", "")
            entries.append({
                "title": title,
                "summary": summary,
                "link": link,
                "published": published
            })

    print(f"Fetched {len(entries)} RSS items.")

    # Filter for crypto-related
    filtered = []
    for it in entries:
        blob = f"{it['title']} {it['summary']}"
        if not is_crypto_related(blob):
            continue

        # Fetch article HTML (fallback to RSS summary)
        try:
            html = fetch(it["link"])
            text = extract_main_text_from_yahoo(html)
            if len(text.split()) < 60:
                text = it["summary"] or it["title"]
        except Exception:
            text = it["summary"] or it["title"]

        # truncate to manageable size for FinBERT
        text = " ".join(text.split()[:1500])
        it["article_text"] = text
        filtered.append(it)
        time.sleep(0.3)

    print(f"Crypto-related items: {len(filtered)}")

    # Sentiment analysis
    rows = []
    print("Running FinBERT sentiment inference...\n")

    for it in filtered:
        res = finbert(it["article_text"])[0]
        label = res["label"]
        score = float(res["score"])
        print(f"[{label:8}] {it['title'][:80]}...")
        rows.append([
            it["title"],
            it["link"],
            it["published"],
            it["summary"],
            label,
            score
        ])

    # Save CSV
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "link", "published", "rss_summary", "finbert_label", "finbert_score"])
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {CSV_FILE}")


# -------------------- ENTRY POINT --------------------

if __name__ == "__main__":
    main()
