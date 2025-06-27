# Digital Memory App

This application helps you analyze your browsing history, identify interesting websites, summarize their content, and track recently closed tabs.

## Features

- Retrieves browsing history from Google Chrome and Microsoft Edge.
- Filters out common and uninteresting domains.
- Summarizes unique URLs using AI (requires internet access).
- Saves interesting sites and their summaries to a CSV file.
- Attempts to identify recently closed tabs from browser history.

## Setup

1.  **Navigate to the project directory:**
    ```bash
    cd /Users/sanzgiri/GeminiProjects/DigitalMemory
    ```

2.  **Create a virtual environment using `uv`:**
    If you don't have `uv` installed, you can install it via pip: `pip install uv`.
    ```bash
    uv venv
    ```

3.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```

4.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

## Usage

To run the application, ensure all target browsers (Chrome, Edge) are completely closed to avoid database locking issues.

```bash
python main.py
```

### Output

- The script will print diagnostic information about the history retrieval process.
- It will output a list of interesting websites and recently closed tabs to the console.
- A CSV file named `interesting_sites.csv` will be created in the project directory, containing the URLs, titles, and AI-generated summaries of interesting websites.

## Troubleshooting

- **`database is locked` error:** Ensure the target browser (Chrome or Edge) is completely closed, including any background processes, before running the script.
- **`ModuleNotFoundError`:** Make sure your virtual environment is activated and all dependencies are installed using `uv pip install -r requirements.txt`.
- **Limited history entries:** The script attempts to join `urls` and `visits` tables for a more comprehensive history. If you still see limited entries, your browser's history might be sparse, or the relevant data is stored in a different format.

## Future Enhancements

- Integration with Safari browsing history.
- Improved detection of recently closed tabs.
- Pattern detection and grouping of websites.
- OneTab integration (requires manual export to HTML or a robust LevelDB reader).
