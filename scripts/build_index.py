"""Generate _site/index.html linking every exported notebook, grouped by topic."""

import html
import pathlib
import re

SITE = pathlib.Path("_site")

# ponytail: topics that import packages missing from Pyodide won't run in-browser
NON_WASM = {"topic08": "Vowpal Wabbit", "topic09": "Prophet"}

topics = {}
for nb in sorted(SITE.glob("topic*/**/index.html")):
    rel = nb.parent.relative_to(SITE)
    topic_dir = rel.parts[0]
    topics.setdefault(topic_dir, []).append(rel)


def note_label(name: str) -> str:
    """notes -> 'notes'; notes-part2-logit-likelihood-learning -> 'part 2: logit likelihood learning'"""
    m = re.match(r"notes-part(\d+)-(.+)", name)
    if m:
        return f"part {m.group(1)}: {m.group(2).replace('-', ' ')}"
    return "notes"


rows = []
for topic_dir, pages in topics.items():
    m = re.match(r"topic(\d+)", topic_dir)
    if m is None:  # non-topic dir slipped into _site; skip it
        continue
    num = m.group(1)
    title = topic_dir.split("-", 1)[1].replace("-", " ").title()
    warn = ""
    if topic_dir[:7] in NON_WASM:
        warn = (
            f" <small>(uses {NON_WASM[topic_dir[:7]]} — may not run in-browser)</small>"
        )
    notes = [p for p in pages if p.parts[-1].startswith("notes")]
    assignments = [p for p in pages if p.parts[-1] == "assignment"]
    subs = []
    if notes:
        links = ", ".join(
            f'<a href="{p}/">{html.escape(note_label(p.parts[-1]))}</a>' for p in notes
        )
        subs.append(f"<li>notes: {links}</li>")
    for p in assignments:
        subs.append(f'<li><a href="{p}/">assignment</a></li>')
    rows.append(
        f"<li><strong>Topic {int(num)}: {html.escape(title)}</strong>{warn}"
        f"<ul>{''.join(subs)}</ul></li>"
    )

SITE.joinpath("index.html").write_text(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mlcourse.ai workbook</title>
<style>
  body {{ font: 16px/1.6 system-ui, sans-serif; max-width: 720px; margin: 3rem auto; padding: 0 1rem; }}
  li {{ margin-bottom: 1rem; }}
  @media (prefers-color-scheme: dark) {{ body {{ background: #111; color: #ddd; }} a {{ color: #8ab4f8; }} }}
</style>
</head>
<body>
<h1>mlcourse.ai workbook</h1>
<p>Notes and assignments for <a href="https://mlcourse.ai">mlcourse.ai</a>, as runnable
<a href="https://marimo.io">marimo</a> notebooks (Python in the browser via WebAssembly).</p>
<ul>
{chr(10).join(rows)}
</ul>
</body>
</html>
""")
print(
    f"index.html: {sum(len(v) for v in topics.values())} notebooks across {len(topics)} topics"
)
