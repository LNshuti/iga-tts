# Batch Trim Design

## Problem

Recording a Kinyarwanda word corpus requires recording many words per session. Currently the app only trims one recording at a time. The user records all words in a single long take and needs to trim each word out, name it by ID, and download all as a zip.

## Solution

Add a "Batch Trim" tab to the existing Gradio app. The user pastes a batch list (with word IDs, Kinyarwanda text, English translations), records one long audio, then steps through each word sequentially to set trim points. At the end, all trimmed files are exported as a zip of `{ID}.wav` files.

## Design

### Batch List Parsing

- Text area accepts the batch format (lines with `ID: \d+`, `Kinyarwanda: ...`, `English: ...`)
- `parse_batch_list(text)` extracts a list of `{"id": int, "kinyarwanda": str, "english": str}` dicts
- Parse button triggers parsing and initializes the wizard

### Sequential Trim Wizard

State (Gradio State):
- `entries`: parsed word list
- `trims`: dict mapping word ID to `(start_time, end_time)` in seconds
- `current_index`: 0-based index into entries

UI:
- Current word display: "Word 3 of 22 -- ID: 142 -- Ivuriro (Hospital)"
- Start/End sliders scoped to full recording duration
- Preview button: plays selected region
- Confirm & Next button: saves trim points, advances index
- Back button: revisit previous word
- Progress indicator

### Export

- Export All button creates a temp directory, writes each trim as `{ID}.wav`
- Zips the directory and returns as Gradio File download
- Enabled once at least one word is confirmed (partial export OK)

### New Functions (all in app.py)

- `parse_batch_list(text)` -- returns list of entry dicts
- `get_current_word_info(entries, index, trims)` -- returns display string with progress
- `confirm_trim(audio, start, end, entries, trims, index)` -- saves trim, advances
- `go_back(entries, trims, index)` -- decrements index
- `preview_trim(audio, start, end)` -- returns trimmed audio tuple for playback
- `export_all(audio, entries, trims)` -- creates zip, returns file path
- `build_batch_tab()` -- constructs the batch trim tab UI

### UI Layout

Existing single-trim UI wrapped in `gr.Tab("Single Trim")`. New batch UI in `gr.Tab("Batch Trim")`. Both inside `gr.Tabs()`.

### Dependencies

No new dependencies. Uses `zipfile` and `os` from stdlib alongside existing `soundfile`, `numpy`, `tempfile`.
