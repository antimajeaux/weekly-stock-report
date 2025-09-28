#!/usr/bin/env python3
"""
worst_performers_report.py

Usage:
- Set environment variables (see README instructions in comments below) or provide them in GitHub Actions secrets.
- Run once or schedule via GitHub Actions (example workflow provided).
"""

import os
import time
import math
import logging
from datetime import datetime, timedelta
import requests
import pandas as pd
import yfinance as yf
import openai
from datetime import datetime, timedelta
import pandas as pd
import requests
from io import StringIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -------------------------
# Configuration (env vars)
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # required
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # or whichever model you use
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # required
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # required (your user chat id)
MARKETCAP_MIN = int(os.getenv("MARKETCAP_MIN", 1_000_000_000))  # 1 billion default
N_US = int(os.getenv("N_US", 20))
N_EU = int(os.getenv("N_EU", 20))
DAYS_BACK = int(os.getenv("DAYS_BACK", 7))  # performance period in days
OPENAI_MAX_RETRIES = 3
SLEEP_BETWEEN_OPENAI = float(os.getenv("SLEEP_BETWEEN_OPENAI", 0.5))  # polite pacing

if not OPENAI_API_KEY or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logging.error("Missing required environment variables. Set OPENAI_API_KEY, TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
    raise SystemExit(1)

openai.api_key = OPENAI_API_KEY

# -------------------------
# Helper functions
# -------------------------
def get_last_week_dates():
    today = datetime.utcnow().date()
    # Saturday run -> go back to Monday of this week
    monday = today - timedelta(days=today.weekday() + 2)  # ensure Saturday goes back to Monday
    friday = monday + timedelta(days=4)
    return monday, friday

MONDAY, FRIDAY = get_last_week_dates()

def fetch_wikipedia_table(url, table_index=0):
    """Return first (or specified) table from a Wikipedia page as DataFrame."""
    logging.info(f"Fetching constituents from {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()  # ensure we notice HTTP errors
    tables = pd.read_html(StringIO(resp.text))
    if table_index >= len(tables):
        raise ValueError("table_index out of range for page tables")
    return tables[table_index]

def get_sp500_tickers():
    # Wikipedia page of S&P 500 constituents
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = fetch_wikipedia_table(url, 0)
    # The ticker column is usually 'Symbol' or 'Ticker symbol'
    for col in ("Symbol", "Ticker symbol", "Ticker"):
        if col in df.columns:
            return df[col].astype(str).str.replace(".", "-", regex=False).tolist()
    raise RuntimeError("Could not find ticker column in S&P 500 table")

def get_stoxx600_tickers():
    # Use STOXX Europe 600 constituents Wikipedia page (constituents table)
    url = "https://en.wikipedia.org/wiki/STOXX_Europe_600"
    # The page usually contains a link or a table; find a table that contains 'Ticker' or 'Company'
    tables = pd.read_html(url)
    # Try to locate the table with tickers
    for t in tables:
        cols = [c.lower() for c in t.columns.astype(str)]
        if any("ticker" in c or "isin" in c or "symbol" in c for c in cols) or any("company" in c for c in cols):
            # try to find ticker-like column
            for c in t.columns:
                if "ticker" in str(c).lower() or "symbol" in str(c).lower() or "isin" in str(c).lower():
                    return t[c].astype(str).str.replace(".", "-", regex=False).tolist()
    # As fallback, try index pages for constituents — but for simplicity raise error so user can review
    raise RuntimeError("Could not find STOXX Europe 600 constituents table automatically. You can supply your own list.")

def get_percent_change_and_marketcap(ticker, days_back=DAYS_BACK+1):
    """
    Returns (percent_change, market_cap, name, last_close, first_close)
    percent_change is (last/first -1) * 100
    """
    try:
        t = yf.Ticker(ticker)
        info = t.get_info()
    except Exception as e:
        logging.debug(f"yfinance info failed for {ticker}: {e}")
        info = {}

    # Some tickers fail; wrap in try
    try:
        hist = t.history(start=str(MONDAY), end=str(FRIDAY + timedelta(days=1)), interval="1d", actions=False)
        # require at least 2 closes
        if hist.shape[0] < 2:
            return None
        closes = hist['Close'].dropna()
        if closes.shape[0] < 2:
            return None
        first_close = float(closes.iloc[0])
        last_close = float(closes.iloc[-1])
        pct = (last_close / first_close - 1.0) * 100.0
    except Exception as e:
        logging.debug(f"yfinance history failed for {ticker}: {e}")
        return None

    market_cap = None
    name = info.get("shortName") or info.get("longName") or ticker
    # try several info keys for market cap
    for key in ("marketCap", "market_cap", "marketCapitalization"):
        if key in info and isinstance(info[key], (int, float)):
            market_cap = int(info[key])
            break

    # In some cases yfinance returns None; try to estimate via sharesOutstanding * previous close
    if market_cap is None:
        shares = info.get("sharesOutstanding")
        if shares:
            market_cap = int(shares * last_close)

    return {
        "ticker": ticker,
        "name": name,
        "pct": pct,
        "market_cap": market_cap if market_cap is not None else 0,
        "first_close": first_close,
        "last_close": last_close
    }

def get_worst_n(tickers, n, marketcap_min):
    results = []
    for i, tk in enumerate(tickers):
        try:
            res = get_percent_change_and_marketcap(tk)
            if not res:
                continue
            if res["market_cap"] >= marketcap_min:
                results.append(res)
        except Exception as e:
            logging.debug(f"Failed ticker {tk}: {e}")
        # small sleep to be polite
        if i % 50 == 0:
            time.sleep(0.5)
    df = pd.DataFrame(results)
    if df.empty:
        return df
    df = df.sort_values("pct")  # worst performers first (more negative)
    return df.head(n)

# -------------------------
# OpenAI prompt
# -------------------------
def make_prompt(company):
    """
    company: dict with ticker, name, pct, market_cap, first_close, last_close
    """
    pct_str = f"{company['pct']:.2f}%"
    mc = company['market_cap']
    mc_readable = f"${mc/1e9:.2f}B" if mc >= 1e9 else f"${mc/1e6:.2f}M"
    prompt = (
        f"You are a concise market analyst.\n\n"
        f"Company: {company['name']} ({company['ticker']})\n"
        f"Market cap: {mc_readable}\n"
        f"1-week performance: {pct_str} (period: last {DAYS_BACK} days)\n\n"
        f"In up to 160 words, list the main plausible reasons this stock fell over the past week. "
        f"Prefer concrete categories (news, earnings, guidance, macro, sector weakness, liquidity, regulatory, company-specific event) and give 3 likely drivers (each 1-2 lines). "
        f"If you don't know specifics, explain what public signals (press, earnings, macro data) one should check to confirm each hypothesis. "
        f"Do NOT invent non-public facts; be explicit about uncertainty."
    )
    return prompt

def ask_openai(prompt, max_tokens=400):
    for attempt in range(1, OPENAI_MAX_RETRIES + 1):
        try:
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            text = resp["choices"][0]["message"]["content"].strip()
            return text
        except Exception as e:
            logging.warning(f"OpenAI call failed (attempt {attempt}): {e}")
            time.sleep(2 ** attempt)
    logging.error("OpenAI failed after retries")
    return "OpenAI call failed — see logs."

def build_markdown_report(us_df, eu_df, analyses):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    md = [f"# Weekly Worst Performers Report\nGenerated: {ts}\nPeriod: last {DAYS_BACK} days\n\n"]
    md.append("## Summary\n")
    md.append(f"- US: worst {len(us_df)} companies (market cap ≥ {MARKETCAP_MIN:,})\n- Europe: worst {len(eu_df)} companies\n\n")
    md.append("## Details\n")
    def df_to_md(df, region):
        s = [f"### {region}\n"]
        for _, row in df.iterrows():
            key = f"{row['ticker']}"
            name = row.get("name", "")
            pct = row.get("pct", 0.0)
            mc = row.get("market_cap", 0)
            mc_read = f"${mc/1e9:.2f}B" if mc >= 1e9 else f"${mc/1e6:.2f}M"
            s.append(f"#### {name} ({key}) — {pct:.2f}% — Market cap: {mc_read}\n")
            analysis = analyses.get(key, "No analysis available.")
            s.append(analysis + "\n\n")
        return "\n".join(s)
    md.append(df_to_md(us_df, "United States"))
    md.append(df_to_md(eu_df, "Europe"))
    return "\n".join(md)

def send_telegram_message(text, token=TELEGRAM_BOT_TOKEN, chat_id=TELEGRAM_CHAT_ID):
    import requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    r = requests.post(url, json=data, timeout=30)
    print(f"Telegram API status: {r.status_code}, response: {r.text}")  # << add this
    r.raise_for_status()
    return r.json()

# -------------------------
# Main flow
# -------------------------
def main():
    logging.info("Starting weekly worst performers report generation")

    # 1) Get tickers
    try:
        us_tickers = get_sp500_tickers()
        logging.info(f"Loaded {len(us_tickers)} US tickers (S&P 500).")
    except Exception as e:
        logging.error(f"Failed to get US tickers: {e}")
        return

    try:
        eu_tickers = get_stoxx600_tickers()
        logging.info(f"Loaded {len(eu_tickers)} European tickers (STOXX Europe constituents).")
    except Exception as e:
        logging.error(f"Failed to get European tickers automatically: {e}")
        logging.error("Aborting. Provide a European ticker list manually if needed.")
        return

    # 2) Calculate worst N
    logging.info("Computing US worst performers...")
    us_worst = get_worst_n(us_tickers, N_US, MARKETCAP_MIN)
    logging.info(f"Found {len(us_worst)} US tickers passing market cap filter.")

    logging.info("Computing EU worst performers...")
    eu_worst = get_worst_n(eu_tickers, N_EU, MARKETCAP_MIN)
    logging.info(f"Found {len(eu_worst)} EU tickers passing market cap filter.")

    # 3) Ask OpenAI for each company
    analyses = {}
    combined = pd.concat([us_worst, eu_worst], ignore_index=True) if not us_worst.empty or not eu_worst.empty else pd.DataFrame()
    for idx, row in combined.iterrows():
        tk = row['ticker']
        prompt = make_prompt(row)
        logging.info(f"Asking OpenAI for analysis of {tk} ({idx+1}/{len(combined)})")
        answer = ask_openai(prompt)
        analyses[tk] = answer
        time.sleep(SLEEP_BETWEEN_OPENAI)

    # 4) Build markdown report
    report_md = build_markdown_report(us_worst, eu_worst, analyses)
    # Telegram has message length limits; we will chunk if too long
    MAX_LEN = 3800  # safe chunk for Telegram Markdown
    logging.info("Sending report to Telegram...")

    # Send a header message first
    header = f"Weekly Worst Performers Report — {datetime.utcnow().strftime('%Y-%m-%d')}\nTotal companies: {len(combined)}"
    send_telegram_message(header)

    # Chunk and send
    chunks = [report_md[i:i+MAX_LEN] for i in range(0, len(report_md), MAX_LEN)]
    for chunk in chunks:
        send_telegram_message(chunk)
        time.sleep(1.0)

    logging.info("Report sent successfully.")

if __name__ == "__main__":
    main()
