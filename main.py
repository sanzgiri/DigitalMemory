import sqlite3
import os
import shutil
import tempfile
import getpass
from urllib.parse import urlparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
import argparse
from datetime import datetime
import re
import json
from typing import Optional, Dict, List
import tomllib

def get_edge_history(target_date: str | None = None):
    """
    Retrieves Microsoft Edge browsing history from the local macOS machine.

    Returns:
        A pandas DataFrame containing the browsing history.
    """
    user = getpass.getuser()
    edge_app_support_path = f"/Users/{user}/Library/Application Support/Microsoft Edge"
    if not os.path.exists(edge_app_support_path):
        print(f"Edge base path not found: {edge_app_support_path}")
        return pd.DataFrame()

    print(f"Scanning Edge profiles in: {edge_app_support_path}")
    all_history = []

    profile_dirs = [d for d in os.listdir(edge_app_support_path)
                    if os.path.isdir(os.path.join(edge_app_support_path, d)) and (d == 'Default' or d.startswith('Profile'))]

    for profile_dir in profile_dirs:
        history_db = os.path.join(edge_app_support_path, profile_dir, 'History')
        if not os.path.exists(history_db):
            continue

        print(f"Attempting to access Edge history database at: {history_db}")

        # Copy to a temporary file to avoid 'database is locked' and open read-only
        tmp_fd, tmp_path = tempfile.mkstemp(prefix='edge_history_', suffix='.db')
        os.close(tmp_fd)
        try:
            shutil.copy2(history_db, tmp_path)
            connection = sqlite3.connect(f"file:{tmp_path}?mode=ro", uri=True)
            cursor = connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"  Tables in '{profile_dir}': {tables}")

            query = """
            SELECT
                u.url,
                u.title,
                datetime(v.visit_time/1000000-11644473600, 'unixepoch', 'localtime') as last_visit_time
            FROM urls u
            JOIN visits v ON u.id = v.url
            WHERE date(datetime(v.visit_time/1000000-11644473600, 'unixepoch', 'localtime')) = ?
            ORDER BY v.visit_time DESC
            """
            params = [target_date] if target_date else [datetime.now().strftime('%Y-%m-%d')]
            df = pd.read_sql_query(query, connection, params=params)
            df['profile'] = profile_dir
            all_history.append(df)
            print(f"  Retrieved {len(df)} rows from Edge profile '{profile_dir}'.")
        except sqlite3.OperationalError as e:
            print(f"  Error accessing database in profile '{profile_dir}': {e}")
            print("  Tip: Ensure Edge is closed, then try again.")
        except Exception as e:
            print(f"  Unexpected error with profile '{profile_dir}': {e}")
        finally:
            try:
                if 'connection' in locals() and connection:
                    connection.close()
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    if all_history:
        return pd.concat(all_history, ignore_index=True)
    else:
        print("No valid Edge history database found in any profile.")
        return pd.DataFrame()

def get_recently_closed_edge_tabs(target_date: str | None = None):
    """
    Retrieves recently closed tabs from Microsoft Edge history database.
    This function looks for specific transition types that might indicate
    a tab being re-opened from a 'recently closed' state.

    Returns:
        A pandas DataFrame containing recently closed tabs.
    """
    user = getpass.getuser()
    edge_app_support_path = f"/Users/{user}/Library/Application Support/Microsoft Edge"
    recently_closed_tabs_df = pd.DataFrame()

    if not os.path.exists(edge_app_support_path):
        return recently_closed_tabs_df

    profile_dirs = [d for d in os.listdir(edge_app_support_path)
                    if os.path.isdir(os.path.join(edge_app_support_path, d)) and (d == 'Default' or d.startswith('Profile'))]

    for profile_dir in profile_dirs:
        history_db = os.path.join(edge_app_support_path, profile_dir, 'History')
        if not os.path.exists(history_db):
            continue

        tmp_fd, tmp_path = tempfile.mkstemp(prefix='edge_history_', suffix='.db')
        os.close(tmp_fd)
        try:
            shutil.copy2(history_db, tmp_path)
            connection = sqlite3.connect(f"file:{tmp_path}?mode=ro", uri=True)
            query = """
            SELECT
                u.url,
                u.title,
                datetime(v.visit_time/1000000-11644473600, 'unixepoch', 'localtime') as last_visit_time,
                v.transition
            FROM urls u
            JOIN visits v ON u.id = v.url
            WHERE v.transition IN (5, 6)
              AND date(datetime(v.visit_time/1000000-11644473600, 'unixepoch', 'localtime')) = ?
            ORDER BY v.visit_time DESC
            LIMIT 50
            """
            params = [target_date] if target_date else [datetime.now().strftime('%Y-%m-%d')]
            temp_df = pd.read_sql_query(query, connection, params=params)
            temp_df['profile'] = profile_dir
            recently_closed_tabs_df = pd.concat([recently_closed_tabs_df, temp_df], ignore_index=True)
        except Exception:
            pass
        finally:
            try:
                if 'connection' in locals() and connection:
                    connection.close()
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    return recently_closed_tabs_df

def get_chrome_history(target_date: str | None = None):
    """
    Retrieves Chrome browsing history from the local macOS machine.

    Returns:
        A pandas DataFrame containing the browsing history.
    """
    user = getpass.getuser()
    chrome_base = f"/Users/{user}/Library/Application Support/Google/Chrome"
    if not os.path.exists(chrome_base):
        print("Chrome base directory not found.")
        return pd.DataFrame()

    profile_dirs = [d for d in os.listdir(chrome_base)
                    if os.path.isdir(os.path.join(chrome_base, d)) and (d == 'Default' or d.startswith('Profile'))]

    all_history = []
    for profile_dir in profile_dirs:
        history_db = os.path.join(chrome_base, profile_dir, 'History')
        if not os.path.exists(history_db):
            continue

        print(f"Attempting to access Chrome history database at: {history_db}")
        tmp_fd, tmp_path = tempfile.mkstemp(prefix='chrome_history_', suffix='.db')
        os.close(tmp_fd)
        try:
            shutil.copy2(history_db, tmp_path)
            connection = sqlite3.connect(f"file:{tmp_path}?mode=ro", uri=True)
            query = """
            SELECT
                u.url,
                u.title,
                datetime(v.visit_time/1000000-11644473600, 'unixepoch', 'localtime') as last_visit_time
            FROM urls u
            JOIN visits v ON u.id = v.url
            WHERE date(datetime(v.visit_time/1000000-11644473600, 'unixepoch', 'localtime')) = ?
            ORDER BY v.visit_time DESC
            """
            params = [target_date] if target_date else [datetime.now().strftime('%Y-%m-%d')]
            df = pd.read_sql_query(query, connection, params=params)
            df['profile'] = profile_dir
            all_history.append(df)
            print(f"  Retrieved {len(df)} rows from Chrome profile '{profile_dir}'.")
        except sqlite3.OperationalError as e:
            print(f"  Error accessing Chrome profile '{profile_dir}': {e}")
            print("  Tip: Ensure Chrome is closed, then try again.")
        except Exception as e:
            print(f"  Unexpected error with Chrome profile '{profile_dir}': {e}")
        finally:
            try:
                if 'connection' in locals() and connection:
                    connection.close()
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    if all_history:
        return pd.concat(all_history, ignore_index=True)
    else:
        print("Chrome history database not found.")
        return pd.DataFrame()

# List of domains to ignore
IGNORED_DOMAINS = [
    'google.com',
    'youtube.com',
    'facebook.com',
    'instagram.com',
    'twitter.com',
    'linkedin.com',
    'amazon.com',
    'netflix.com',
    'spotify.com',
    'apple.com',
    'microsoft.com',
    'en.wikipedia.org',
]

def filter_history(history_df, ignored_domains):
    """
    Filters the browsing history to exclude common domains.

    Args:
        history_df: A pandas DataFrame with the browsing history.
        ignored_domains: A list of domains to ignore.

    Returns:
        A pandas DataFrame with the filtered history.
    """
    if history_df.empty:
        return history_df

    # Extract domain robustly (supports http/https and no trailing slash)
    def extract_domain(u: str):
        try:
            return urlparse(u).netloc.lower()
        except Exception:
            return ''

    history_df = history_df.copy()
    history_df['domain'] = history_df['url'].apply(extract_domain)
    filtered_history = history_df[~history_df['domain'].isin([d.lower() for d in ignored_domains])]
    return filtered_history

# --- LLM (Ollama) integration helpers ---
def call_ollama(model: str, prompt: str, host: str = "http://localhost:11434", timeout: int = 20) -> Optional[str]:
    try:
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response")
    except Exception:
        return None

def extract_page_metadata(url: str) -> Dict[str, str]:
    result = {"title": "", "description": ""}
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
        }
        r = requests.get(url, headers=headers, timeout=8)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'lxml')
        title_tag = soup.find('title')
        if title_tag:
            result["title"] = title_tag.get_text(strip=True)
        meta_desc = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
        if meta_desc and meta_desc.get('content'):
            result["description"] = meta_desc.get('content').strip()
        else:
            p = soup.find('p')
            if p:
                result["description"] = p.get_text(strip=True)[:400]
    except Exception:
        pass
    return result

def load_rules(rules_path: Optional[str] = None) -> Dict:
    # Default rules if no file provided
    default_rules = {
        "exclude": {
            "email": [r"mail\.google\.com", r"outlook\.live\.com", r"mail\.yahoo\.com", r"icloud\.com/mail"],
            "social": [r"facebook\.com", r"instagram\.com", r"twitter\.com|x\.com", r"linkedin\.com", r"tiktok\.com"],
            "finance": [r"chase\.com", r"bankofamerica\.com", r"wellsfargo\.com", r"schwab\.com", r"robinhood\.com", r"fidelity\.com"],
            "news": [r"nytimes\.com", r"cnn\.com", r"bbc\.co\.uk|bbc\.com", r"foxnews\.com", r"washingtonpost\.com", r"wsj\.com", r"reuters\.com"],
            "streaming": [r"netflix\.com", r"youtube\.com", r"spotify\.com", r"hulu\.com", r"disneyplus\.com"],
            "shopping": [r"amazon\.com", r"ebay\.com", r"etsy\.com"],
        },
        "informational_keywords": ["docs", "wiki", "guide", "how to", "research", "paper", "api", "reference", "tutorial", "documentation"],
    }
    path = rules_path or "rules.toml"
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                data = tomllib.load(f)
            # Shallow merge
            if 'exclude' in data:
                for k, v in data['exclude'].items():
                    default_rules['exclude'][k] = v
            if 'informational_keywords' in data:
                default_rules['informational_keywords'] = data['informational_keywords']
        except Exception:
            pass
    return default_rules

def classify_with_rules(domain: str, titles: List[str], rules: Optional[Dict] = None) -> Dict[str, str | bool]:
    d = domain.lower()
    title_blob = " ".join(titles).lower()[:1000]
    rules = rules or load_rules()

    # Exclusions
    for cat, pats in rules.get('exclude', {}).items():
        for p in pats:
            try:
                if re.search(p, d):
                    return {"rule_category": f"exclude_{cat}", "rule_interesting": False}
            except re.error:
                continue

    # Heuristic informational
    info_keywords = [kw.lower() for kw in rules.get('informational_keywords', [])]
    informational = any(k in title_blob for k in info_keywords)
    return {"rule_category": "informational" if informational else "other", "rule_interesting": informational}

def classify_with_llm(domain: str, sample_titles: List[str], sample_urls: List[str], model: str, host: str) -> Optional[bool]:
    prompt = (
        "You are a classifier. Determine if the following domain represents informational content (guides, documentation, articles, tutorials, research) "
        "and is not news (national/world), personal email, social media, or personal finance/banking.\n\n"
        f"Domain: {domain}\n"
        f"Sample titles: {json.dumps(sample_titles[:3], ensure_ascii=False)}\n"
        f"Sample URLs: {json.dumps(sample_urls[:3], ensure_ascii=False)}\n\n"
        "Respond with a single word: YES if informational, NO otherwise."
    )
    resp = call_ollama(model=model, prompt=prompt, host=host)
    if not resp:
        return None
    text = resp.strip().upper()
    if "YES" in text and "NO" not in text:
        return True
    if "NO" in text and "YES" not in text:
        return False
    return None

def consolidate_by_domain(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Expect columns: url, title, last_visit_time, domain
    def agg_titles(series):
        return [t for t in series.dropna().astype(str).head(5)]
    def agg_urls(series):
        return [u for u in series.dropna().astype(str).head(5)]
    grouped = df.groupby('domain').agg(
        visit_count=pd.NamedAgg(column='url', aggfunc='count'),
        last_visit_time=pd.NamedAgg(column='last_visit_time', aggfunc='max'),
        sample_title=pd.NamedAgg(column='title', aggfunc='first'),
        sample_url=pd.NamedAgg(column='url', aggfunc='first'),
        all_titles=pd.NamedAgg(column='title', aggfunc=agg_titles),
        all_urls=pd.NamedAgg(column='url', aggfunc=agg_urls),
    ).reset_index()
    return grouped

def summarize_url(url: str) -> str:
    """
    Summarizes the content of a given URL using the web_fetch tool.

    Args:
        url: The URL to summarize.

    Returns:
        A string containing the summary, or an error message if summarization fails.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'lxml')

        # Prefer meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
        if meta_desc and meta_desc.get('content'):
            desc = meta_desc.get('content').strip()
        else:
            # Fallback: first meaningful paragraph
            p = soup.find('p')
            desc = p.get_text(strip=True) if p else ''

        title_tag = soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else url
        if desc:
            return f"{title} — {desc[:300]}"
        return title
    except Exception as e:
        return f"(summary unavailable: {e})"

def summarize_with_llm(domain_row: pd.Series, model: str, host: str) -> Optional[str]:
    titles = domain_row.get('all_titles') or []
    urls = domain_row.get('all_urls') or []
    primary_url = domain_row.get('sample_url')
    meta = extract_page_metadata(primary_url) if isinstance(primary_url, str) else {"title": "", "description": ""}
    prompt = (
        "Summarize concisely (1-2 sentences) what this website is about for personal knowledge tracking.\n"
        f"Domain: {domain_row['domain']}\n"
        f"Sample titles: {json.dumps(titles[:3], ensure_ascii=False)}\n"
        f"Page title: {meta.get('title','')}\n"
        f"Description: {meta.get('description','')}\n"
    )
    resp = call_ollama(model=model, prompt=prompt, host=host)
    return resp.strip() if resp else None

def ollama_available(host: str) -> bool:
    try:
        r = requests.get(f"{host}/api/tags", timeout=3)
        return r.ok
    except Exception:
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Digital Memory daily history summarizer')
    parser.add_argument('--date', help="Target date in YYYY-MM-DD (defaults to today)")
    parser.add_argument('--llm-classify', action='store_true', help='Use LLM (Ollama) to classify informational domains')
    parser.add_argument('--llm-summaries', action='store_true', help='Use LLM (Ollama) to produce summaries')
    parser.add_argument('--ollama-model', default='mistral', help='Ollama model name (default: mistral)')
    parser.add_argument('--ollama-host', default='http://localhost:11434', help='Ollama host URL')
    parser.add_argument('--rules', default='rules.toml', help='Path to rules TOML file (optional)')
    args = parser.parse_args()

    target_date = args.date if args.date else datetime.now().strftime('%Y-%m-%d')

    # Preflight: if LLM flags are on but Ollama isn't reachable, disable LLM features gracefully
    if (args.llm_classify or args.llm_summaries) and not ollama_available(args.ollama_host):
        print(f"Warning: Ollama not available at {args.ollama_host}. Disabling LLM features for this run.")
        args.llm_classify = False
        args.llm_summaries = False

    history_df = pd.DataFrame()

    # Try to get Chrome history first
    history_df = get_chrome_history(target_date)

    # If Chrome history is empty, try Edge history
    if history_df.empty:
        history_df = get_edge_history(target_date)

    if history_df.empty:
        print("No browser history found from Chrome or Edge.")
        raise SystemExit(0)

    print("Successfully retrieved browser history.")
    print(f"Found {len(history_df)} total entries.")

    # Filter obvious non-interesting domains list, then consolidate by domain
    filtered = filter_history(history_df, IGNORED_DOMAINS)
    domains_df = consolidate_by_domain(filtered)
    print(f"Consolidated into {len(domains_df)} domains.")

    # Rule-based classification
    rules = load_rules(args.rules)
    rule_results = domains_df.apply(lambda row: classify_with_rules(row['domain'], row.get('all_titles', []), rules=rules), axis=1)
    rule_df = pd.DataFrame(list(rule_results))
    domains_df = pd.concat([domains_df, rule_df], axis=1)

    # Optional LLM classification
    if args.llm_classify:
        llm_flags: List[Optional[bool]] = []
        for _, row in domains_df.iterrows():
            llm_flag = classify_with_llm(row['domain'], row.get('all_titles', []), row.get('all_urls', []), args.ollama_model, args.ollama_host)
            llm_flags.append(llm_flag)
        domains_df['llm_informational'] = llm_flags
    else:
        domains_df['llm_informational'] = None

    # Decide interesting: rule_interesting OR llm_informational==True, but never include excluded categories
    def final_interest(row):
        if isinstance(row.get('rule_category'), str) and row['rule_category'].startswith('exclude_'):
            return False
        if row.get('llm_informational') is True:
            return True
        return bool(row.get('rule_interesting'))

    domains_df['interesting'] = domains_df.apply(final_interest, axis=1)
    interesting_domains = domains_df[domains_df['interesting']].copy()
    print(f"Selected {len(interesting_domains)} interesting domains.")

    # Summaries (LLM optional, fallback to HTML-based)
    summaries: List[str] = []
    for _, row in interesting_domains.iterrows():
        summary = None
        if args.llm_summaries:
            summary = summarize_with_llm(row, args.ollama_model, args.ollama_host)
        if not summary:
            # Fallback: summarize the sample_url with HTML extraction
            sample_url = row.get('sample_url')
            summary = summarize_url(sample_url) if isinstance(sample_url, str) else ''
        summaries.append(summary)
    interesting_domains['summary'] = summaries

    # Output CSV
    output_cols = [
        'domain', 'visit_count', 'last_visit_time', 'sample_url', 'sample_title',
        'rule_category', 'rule_interesting', 'llm_informational', 'interesting', 'summary'
    ]
    missing = [c for c in output_cols if c not in interesting_domains.columns]
    for c in missing:
        interesting_domains[c] = None

    output_csv_path = f"interesting_sites_{target_date}.csv"
    interesting_domains[output_cols].to_csv(output_csv_path, index=False)
    print(f"Summaries saved to {output_csv_path}")
