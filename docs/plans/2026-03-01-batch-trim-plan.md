# Batch Trim Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a batch trim tab so users can record all words in one take, paste a batch list, step through each word to set trim points, and download all trimmed files as a zip.

**Architecture:** New tab in existing `app.py`. Sequential wizard pattern using Gradio State for entries/trims/index. All functions added to the single file. Export produces a zip of `{ID}.wav` files.

**Tech Stack:** Gradio (existing), soundfile (existing), zipfile + os (stdlib)

---

### Task 1: Add `parse_batch_list` function

**Files:**
- Modify: `app.py` (add after `update_selection_info`, around line 89)

**Step 1: Write `parse_batch_list`**

Add this function after `update_selection_info`:

```python
import re


def parse_batch_list(text):
    """Parse a batch list text into a list of word entries.

    Expected format has lines like:
        ID: 140
        Kinyarwanda: Inzu
        English: House

    Returns:
        Tuple of (entries list, status message)
    """
    if not text or not text.strip():
        return [], "No batch list provided"

    entries = []
    current = {}

    for line in text.strip().split("\n"):
        line = line.strip()

        id_match = re.match(r"ID:\s*(\d+)", line)
        if id_match:
            current["id"] = int(id_match.group(1))
            continue

        kiny_match = re.match(r"Kinyarwanda:\s*(.+)", line)
        if kiny_match:
            current["kinyarwanda"] = kiny_match.group(1).strip()
            continue

        eng_match = re.match(r"English:\s*(.+)", line)
        if eng_match:
            current["english"] = eng_match.group(1).strip()
            if "id" in current and "kinyarwanda" in current:
                entries.append(dict(current))
                current = {}
            continue

    if not entries:
        return [], "Could not parse any entries. Check format."

    return entries, f"Parsed {len(entries)} words"
```

**Step 2: Verify manually**

Run: `python -c "from app import parse_batch_list; entries, msg = parse_batch_list('ID: 140\nKinyarwanda: Inzu\nEnglish: House\n\nID: 141\nKinyarwanda: Ishuri\nEnglish: School'); print(entries, msg)"`

Expected: Two entry dicts printed, message says "Parsed 2 words"

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add parse_batch_list function"
```

---

### Task 2: Add batch trim helper functions

**Files:**
- Modify: `app.py` (add after `parse_batch_list`)

**Step 1: Write helper functions**

```python
import zipfile
import os


def get_current_word_display(entries, current_index, trims):
    """Return display string for the current word in the wizard."""
    if not entries:
        return "No words loaded. Paste a batch list and click Parse."

    if current_index >= len(entries):
        confirmed = len(trims)
        return f"All done! {confirmed}/{len(entries)} words trimmed. Click Export."

    entry = entries[current_index]
    confirmed = len(trims)
    total = len(entries)
    trimmed_mark = " [TRIMMED]" if entry["id"] in trims else ""

    return (
        f"**Word {current_index + 1} of {total}** (confirmed: {confirmed}) — "
        f"ID: {entry['id']} — {entry['kinyarwanda']} ({entry['english']}){trimmed_mark}"
    )


def confirm_and_next(audio, start_time, end_time, entries, trims, current_index):
    """Save trim points for current word, advance to next."""
    if audio is None or not entries:
        return trims, current_index, "No audio or entries loaded"

    if current_index >= len(entries):
        return trims, current_index, get_current_word_display(entries, current_index, trims)

    entry = entries[current_index]
    trims[entry["id"]] = (start_time, end_time)

    new_index = current_index + 1
    return trims, new_index, get_current_word_display(entries, new_index, trims)


def go_back(entries, trims, current_index):
    """Go back to previous word."""
    if current_index <= 0:
        return current_index, get_current_word_display(entries, 0, trims)

    new_index = current_index - 1
    entry = entries[new_index]

    # Restore previous trim points as slider values if they exist
    if entry["id"] in trims:
        start, end = trims[entry["id"]]
        return new_index, get_current_word_display(entries, new_index, trims), start, end

    return new_index, get_current_word_display(entries, new_index, trims), None, None


def preview_trim(audio, start_time, end_time):
    """Return trimmed audio for playback preview."""
    if audio is None:
        return None

    sample_rate, audio_data = audio
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    if start_sample >= end_sample:
        return None

    return (sample_rate, audio_data[start_sample:end_sample])


def export_all(audio, entries, trims):
    """Export all confirmed trims as a zip of {ID}.wav files."""
    if audio is None or not entries or not trims:
        return None

    sample_rate, audio_data = audio
    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, "trimmed_words.zip")

    with zipfile.ZipFile(zip_path, "w") as zf:
        for entry in entries:
            word_id = entry["id"]
            if word_id not in trims:
                continue

            start_time, end_time = trims[word_id]
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            if start_sample >= end_sample:
                continue

            trimmed = audio_data[start_sample:end_sample]
            wav_path = os.path.join(tmp_dir, f"{word_id}.wav")
            sf.write(wav_path, trimmed, sample_rate)
            zf.write(wav_path, f"{word_id}.wav")

    return zip_path
```

**Step 2: Commit**

```bash
git add app.py
git commit -m "feat: add batch trim helper functions"
```

---

### Task 3: Build the batch trim tab UI

**Files:**
- Modify: `app.py` — restructure `build_ui()` to use tabs, add `build_batch_tab()`

**Step 1: Wrap existing UI in a tab and add batch tab**

Restructure `build_ui()`:
- Add `import re, zipfile, os` at top of file (alongside existing imports)
- Wrap existing UI content inside `gr.Tab("Single Trim")`
- Add new `gr.Tab("Batch Trim")` with the wizard UI

The batch tab layout:

```python
with gr.Tab("Batch Trim"):
    gr.Markdown("### Batch Recording & Trimming")
    gr.Markdown("Record all words in one take, paste your batch list, then trim each word.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### 1. Paste Batch List")
            batch_text = gr.Textbox(
                label="Batch List",
                placeholder="Paste your batch list here...",
                lines=10
            )
            parse_button = gr.Button("Parse List", variant="secondary")
            parse_status = gr.Textbox(label="Parse Status", interactive=False)

            gr.Markdown("#### 2. Record All Words")
            batch_audio = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Record All Words",
                buttons=[]
            )
            batch_audio_info = gr.Textbox(
                value="No audio recorded yet",
                label="Recording Info",
                interactive=False
            )

        with gr.Column(scale=1):
            gr.Markdown("#### 3. Trim Each Word")
            word_display = gr.Markdown("No words loaded. Paste a batch list and click Parse.")

            batch_start = gr.Slider(
                minimum=0, maximum=1, value=0, step=0.01,
                label="Start Time (seconds)", interactive=False
            )
            batch_end = gr.Slider(
                minimum=0, maximum=1, value=1, step=0.01,
                label="End Time (seconds)", interactive=False
            )
            batch_selection = gr.Textbox(
                value="No audio recorded yet",
                label="Selection", interactive=False
            )

            preview_audio = gr.Audio(label="Preview", type="numpy")

            with gr.Row():
                back_button = gr.Button("Back", variant="secondary")
                preview_button = gr.Button("Preview", variant="secondary")
                confirm_button = gr.Button("Confirm & Next", variant="primary")

            gr.Markdown("#### 4. Export")
            export_button = gr.Button("Export All as Zip", variant="primary", size="lg")
            zip_output = gr.File(label="Download Zip")

    # Gradio State
    entries_state = gr.State([])
    trims_state = gr.State({})
    index_state = gr.State(0)
```

**Step 2: Wire event handlers**

```python
    # Parse batch list
    def on_parse(text):
        entries, msg = parse_batch_list(text)
        display = get_current_word_display(entries, 0, {})
        return entries, {}, 0, msg, display

    parse_button.click(
        fn=on_parse,
        inputs=[batch_text],
        outputs=[entries_state, trims_state, index_state, parse_status, word_display]
    )

    # Update sliders when audio changes
    def on_batch_audio_change(audio):
        if audio is None:
            return (
                gr.Slider(maximum=1, value=0, interactive=False),
                gr.Slider(maximum=1, value=1, interactive=False),
                "No audio recorded yet"
            )
        sample_rate, audio_data = audio
        duration = len(audio_data) / sample_rate
        return (
            gr.Slider(maximum=duration, value=0, interactive=True),
            gr.Slider(maximum=duration, value=duration, interactive=True),
            f"Total duration: {duration:.2f}s"
        )

    batch_audio.change(
        fn=on_batch_audio_change,
        inputs=[batch_audio],
        outputs=[batch_start, batch_end, batch_audio_info]
    )

    # Selection info
    batch_start.change(
        fn=update_selection_info,
        inputs=[batch_audio, batch_start, batch_end],
        outputs=[batch_selection]
    )
    batch_end.change(
        fn=update_selection_info,
        inputs=[batch_audio, batch_start, batch_end],
        outputs=[batch_selection]
    )

    # Preview
    preview_button.click(
        fn=preview_trim,
        inputs=[batch_audio, batch_start, batch_end],
        outputs=[preview_audio]
    )

    # Confirm & Next
    def on_confirm(audio, start, end, entries, trims, idx):
        trims, new_idx, display = confirm_and_next(audio, start, end, entries, trims, idx)
        return trims, new_idx, display

    confirm_button.click(
        fn=on_confirm,
        inputs=[batch_audio, batch_start, batch_end, entries_state, trims_state, index_state],
        outputs=[trims_state, index_state, word_display]
    )

    # Back
    def on_back(entries, trims, idx):
        result = go_back(entries, trims, idx)
        if len(result) == 4:
            new_idx, display, start, end = result
            return new_idx, display, gr.Slider(value=start), gr.Slider(value=end)
        new_idx, display, _, _ = result
        return new_idx, display, gr.Slider(), gr.Slider()

    back_button.click(
        fn=on_back,
        inputs=[entries_state, trims_state, index_state],
        outputs=[index_state, word_display, batch_start, batch_end]
    )

    # Export
    export_button.click(
        fn=export_all,
        inputs=[batch_audio, entries_state, trims_state],
        outputs=[zip_output]
    )
```

**Step 3: Run the app and test manually**

Run: `python app.py`

Test:
1. Both tabs appear
2. Single Trim tab works as before
3. Paste sample batch list, click Parse, see "Parsed 2 words"
4. Record audio, trim wizard works
5. Export produces a zip

**Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add batch trim tab with sequential wizard and zip export"
```

---

### Task 4: Final integration test

**Files:**
- No file changes, manual testing

**Step 1: Run full workflow test**

Run: `python app.py`

Full test with the 22-word batch list from the user's test case. Verify:
1. Paste full batch list -> "Parsed 22 words"
2. Record audio
3. Step through a few words, confirm trims
4. Go back, re-trim
5. Export zip, verify it contains `{ID}.wav` files

**Step 2: Commit any fixes if needed**

---

## Summary

| Task | Description | Key Files |
|------|------------|-----------|
| 1 | `parse_batch_list` function | `app.py` |
| 2 | Helper functions (confirm, back, preview, export) | `app.py` |
| 3 | Batch trim tab UI + event wiring | `app.py` |
| 4 | Integration test | manual |
