# Instructions

## Reading the workbook

Browse it live at **[kidskoding.github.io/mlcourse-ai-workbook](https://kidskoding.github.io/mlcourse-ai-workbook/)**. Every notebook runs Python in your browser via WebAssembly — you can read the notes, run the code, and tweak cells right on the page.

> [!NOTE]
> The live site is read-only: edits run in your browser tab but are gone on refresh — nothing saves back. Topics 8–9 import packages (Vowpal Wabbit, Prophet) that don't exist in the browser runtime; run those locally.

## Doing the assignments

Work happens locally, in your own copy. The only prerequisite is [uv](https://docs.astral.sh/uv/) — marimo and all dependencies come from the lockfile.

1. Fork this repo on GitHub, then:

```sh
git clone https://github.com/<your-username>/mlcourse-ai-workbook
cd mlcourse-ai-workbook
uv sync
uv run marimo edit --watch .
```

2. A browser tab opens listing every notebook. Each topic directory has `notes*.py` (study notes; multi-part topics have one file per course page) and an `assignment.py` with `# TODO` cells to fill in.

3. Fill in the `# TODO` cells — marimo saves straight to the `.py` files. Commit and push; your fork is your progress.

`--watch` makes marimo pick up edits made outside the editor (e.g. by an agent or another editor).

## Other ways to run

Jump straight into one notebook:

```sh
uv run marimo edit --watch topic01-exploratory-data-analysis-pandas/notes.py
```

Run a notebook end-to-end as a plain script:

```sh
uv run python topic01-exploratory-data-analysis-pandas/notes.py
```
