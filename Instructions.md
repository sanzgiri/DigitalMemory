# Digital Memory App - Instructions

This document summarizes the instructions provided for building the "Digital Memory" application.

## Core Functionality

The primary goal is to create an application that:
- Goes through your browsing history (initially Google Chrome, then Microsoft Edge, and eventually Safari).
- Groups websites to detect patterns.
- Saves summaries of "non-standard" websites (e.g., not email, financial sites).
- Looks for open tabs (specifically mentioned OneTab and Recently Closed entries).
- Records interesting websites with brief summaries.

## Implementation Details & Progress

### Application Type
- A Python Command-Line Interface (CLI) application.

### Project Structure
- All project files are to be located in a dedicated directory: `/Users/sanzgiri/GeminiProjects/DigitalMemory`.

### Environment Management
- Use `uv` for environment and package management for this and all future projects.

### Browsing History Extraction
- **Google Chrome:** Functionality to extract history from Chrome's SQLite database is implemented.
- **Microsoft Edge:**
    - Functionality to extract history from Edge's SQLite database is implemented.
    - The script iterates through all detected Edge profiles to find the active history database.
    - Initial attempts to read the `History` file encountered "no such table: urls" errors, which were debugged to reveal that the `Default` profile's `History` file was being locked by the browser or was empty.
    - The script now attempts to join `urls` and `visits` tables for a more comprehensive history.
    - Debugging output for table schemas and row counts was added to assist in understanding the database structure.

### AI-Powered Summarization
- A `summarize_url` function is implemented, which uses the `web_fetch` tool to generate summaries of URLs.

### Local Data Storage
- The application saves the filtered and summarized "interesting" history to a CSV file named `interesting_sites.csv`.

### OneTab Integration
- Initial attempts to read OneTab data directly from `extension://` URLs were unsuccessful due to tool limitations.
- Investigation into OneTab's local storage revealed it uses LevelDB.
- Attempts to install `plyvel` (a Python LevelDB library) encountered compilation issues related to missing LevelDB C++ headers, which were not resolved.
- **Current Status:** OneTab integration is paused due to `plyvel` installation difficulties. Manual export to HTML is the suggested alternative if OneTab data is critical.

### Recently Closed Tabs
- Functionality to retrieve "Recently Closed" tabs from Microsoft Edge's history database is implemented.
- This function queries the `visits` table for specific `transition` types (e.g., `5` for `form_submit` and `6` for `reload`) that might indicate re-opened tabs.
- The results of this function are printed separately for analysis.

## Current Checkpoint

The application can:
- Retrieve browsing history from Chrome (if available) or Edge.
- Filter out common domains.
- Summarize interesting URLs using `web_fetch`.
- Save the results to `interesting_sites.csv`.
- Attempt to retrieve and display recently closed Edge tabs (based on `transition` types).

## Next Steps (as per user request)

1.  **Create `README.md`:** Provide instructions on how to set up and run the application.
2.  **Check in code to Git:** Initialize a Git repository and commit the current state of the code.

## Future Enhancements (from original plan)

- Integrate Safari browsing history.
- Refine "Recently Closed" tab detection.
- Implement pattern detection and grouping of websites.
- Further explore OneTab integration if a viable method is found.
