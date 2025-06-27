import sqlite3
import os
import getpass
import pandas as pd

def get_edge_history():
    """
    Retrieves Microsoft Edge browsing history from the local macOS machine.
    It iterates through all detected profiles to find the active history.

    Returns:
        A pandas DataFrame containing the browsing history.
    """
    user = getpass.getuser()
    edge_app_support_path = f'/Users/{user}/Library/Application Support/Microsoft Edge'

    if not os.path.exists(edge_app_support_path):
        print(f"Microsoft Edge application support path not found at: {edge_app_support_path}")
        return pd.DataFrame()

    profile_dirs = [d for d in os.listdir(edge_app_support_path) if os.path.isdir(os.path.join(edge_app_support_path, d))]

    for profile_dir in profile_dirs:
        history_db = os.path.join(edge_app_support_path, profile_dir, 'History')
        print(f"Attempting to access Edge history database in profile '{profile_dir}' at: {history_db}")

        if not os.path.exists(history_db):
            print(f"  History file not found in profile '{profile_dir}'. Skipping.")
            continue

        try:
            connection = sqlite3.connect(history_db)
            cursor = connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"  Tables found in profile '{profile_dir}' database: {tables}")

            if not tables:
                print(f"  Warning: Database in profile '{profile_dir}' exists but contains no tables. Skipping.")
                connection.close()
                continue

            # Inspect schema of relevant tables
            for table_name in ['urls', 'visits', 'edge_urls', 'edge_visits']:
                if (table_name,) in tables:
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    columns = cursor.fetchall()
                    print(f"  Schema for table '{table_name}': {columns}")

            # Get row counts for urls and visits tables
            if ('urls',) in tables:
                cursor.execute("SELECT COUNT(*) FROM urls;")
                urls_count = cursor.fetchone()[0]
                print(f"  Number of entries in 'urls' table: {urls_count}")
            if ('visits',) in tables:
                cursor.execute("SELECT COUNT(*) FROM visits;")
                visits_count = cursor.fetchone()[0]
                print(f"  Number of entries in 'visits' table: {visits_count}")

            query = """
            SELECT
                u.url,
                u.title,
                datetime(v.visit_time/1000000-11644473600, 'unixepoch', 'localtime') as last_visit_time
            FROM urls u
            JOIN visits v ON u.id = v.url
            ORDER BY v.visit_time DESC
            """
            history_df = pd.read_sql_query(query, connection)
            print(f"Successfully retrieved history from profile '{profile_dir}'.")
            return history_df

        except sqlite3.OperationalError as e:
            print(f"  Error accessing database in profile '{profile_dir}': {e}")
            print("  Please make sure Edge is closed before running this script.")
        except Exception as e:
            print(f"  An unexpected error occurred with profile '{profile_dir}': {e}")
        finally:
            if 'connection' in locals() and connection:
                connection.close()

    print("No valid Edge history database found across all profiles.")
    return pd.DataFrame()

def get_recently_closed_edge_tabs():
    """
    Retrieves recently closed tabs from Microsoft Edge history database.
    This function looks for specific transition types that might indicate
    a tab being re-opened from a 'recently closed' state.

    Returns:
        A pandas DataFrame containing recently closed tabs.
    """
    user = getpass.getuser()
    edge_app_support_path = f'/Users/{user}/Library/Application Support/Microsoft Edge'
    recently_closed_tabs_df = pd.DataFrame()

    profile_dirs = [d for d in os.listdir(edge_app_support_path) if os.path.isdir(os.path.join(edge_app_support_path, d))]

    for profile_dir in profile_dirs:
        history_db = os.path.join(edge_app_support_path, profile_dir, 'History')
        if not os.path.exists(history_db):
            continue

        try:
            connection = sqlite3.connect(history_db)
            query = """
            SELECT
                u.url,
                u.title,
                datetime(v.visit_time/1000000-11644473600, 'unixepoch', 'localtime') as last_visit_time,
                v.transition
            FROM urls u
            JOIN visits v ON u.id = v.url
            WHERE v.transition IN (5, 6) -- Transition types for 'form_submit' and 'reload' which can indicate re-opened tabs
            ORDER BY v.visit_time DESC
            LIMIT 50 -- Limit to recent entries
            """
            temp_df = pd.read_sql_query(query, connection)
            recently_closed_tabs_df = pd.concat([recently_closed_tabs_df, temp_df], ignore_index=True)

        except sqlite3.OperationalError as e:
            # print(f"  Error accessing database for recently closed tabs in profile '{profile_dir}': {e}")
            pass # Suppress error for profiles without history
        except Exception as e:
            # print(f"  An unexpected error occurred with recently closed tabs in profile '{profile_dir}': {e}")
            pass
        finally:
            if 'connection' in locals() and connection:
                connection.close()

    return recently_closed_tabs_df

def get_chrome_history():
    """
    Retrieves Chrome browsing history from the local macOS machine.

    Returns:
        A pandas DataFrame containing the browsing history.
    """
    # Get the current username
    user = getpass.getuser()

    # Path to the Chrome history database
    history_db = f'/Users/{user}/Library/Application Support/Google/Chrome/Default/History'

    if not os.path.exists(history_db):
        print("Chrome history database not found.")
        return pd.DataFrame()

    try:
        # Connect to the database
        connection = sqlite3.connect(history_db)

        # Query the database
        query = """
        SELECT url, title,
               datetime(last_visit_time/1000000-11644473600, 'unixepoch', 'localtime') as last_visit_time
        FROM urls
        ORDER BY last_visit_time DESC
        """

        # Read the data into a pandas DataFrame
        history_df = pd.read_sql_query(query, connection)

        return history_df

    except sqlite3.OperationalError as e:
        print(f"Error accessing the database: {e}")
        print("Please make sure Chrome is closed before running this script.")
        return pd.DataFrame()
    finally:
        # Close the connection
        if 'connection' in locals() and connection:
            connection.close()

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
    # Extract the domain from the URL
    history_df['domain'] = history_df['url'].str.extract(r'https:\/\/([^\/]+)\/')

    # Filter out the ignored domains
    filtered_history = history_df[~history_df['domain'].isin(ignored_domains)]

    return filtered_history

def summarize_url(url: str) -> str:
    """
    Summarizes the content of a given URL using the web_fetch tool.

    Args:
        url: The URL to summarize.

    Returns:
        A string containing the summary, or an error message if summarization fails.
    """
    try:
        # Use web_fetch to get the content and summarize it
        response = default_api.web_fetch(prompt=f"Summarize the content of {url}")
        if response and 'output' in response:
            return response['output']
        else:
            return "Could not summarize URL: No output from web_fetch."
    except Exception as e:
        return f"Error summarizing URL {url}: {e}"

if __name__ == '__main__':
    history_df = pd.DataFrame()

    # Try to get Chrome history first
    history_df = get_chrome_history()

    # If Chrome history is empty, try Edge history
    if history_df.empty:
        history_df = get_edge_history()

    if not history_df.empty:
        print("Successfully retrieved browser history.")
        print(f"Found {len(history_df)} total entries.")

        # Filter the history
        interesting_history = filter_history(history_df, IGNORED_DOMAINS)

        print(f"Found {len(interesting_history)} interesting entries after filtering.")
        print("Here are the 10 most recent interesting entries:")
        print(interesting_history.head(10))

        # Get and display recently closed tabs
        print("\nAttempting to retrieve recently closed Edge tabs...")
        recently_closed_tabs = get_recently_closed_edge_tabs()
        if not recently_closed_tabs.empty:
            print(f"Found {len(recently_closed_tabs)} recently closed tabs.")
            print("Here are the 10 most recent recently closed tabs:")
            print(recently_closed_tabs.head(10))
        else:
            print("No recently closed Edge tabs found.")

        # Add summarization and save to CSV
        print("\nSummarizing interesting URLs and saving to CSV...")
        interesting_history['summary'] = interesting_history['url'].apply(summarize_url)

        output_csv_path = "interesting_sites.csv"
        interesting_history.to_csv(output_csv_path, index=False)
        print(f"Summaries saved to {output_csv_path}")

    else:
        print("No browser history found from Chrome or Edge.")
