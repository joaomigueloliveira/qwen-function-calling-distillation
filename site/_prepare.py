"""Pre-render step, generating what the site needs from the repo (neither is committed):
- docs/adr/*.md become the Decisions pages (decisions/);
- the figures' data (docs/figures/*.json) and the probe (data/probe/*.json) go to assets/.

Each ADR is rendered as written. The first heading becomes the page title, links into the repo
point at GitHub, `#123` becomes a link to that issue, and "ADR 0006" links to its page.
"""

import re
import shutil
from pathlib import Path

REPO = "https://github.com/joaomigueloliveira/qwen-function-calling-distillation"
SITE = Path(__file__).parent
ADRS = SITE.parent / "docs" / "adr"
OUT = SITE / "decisions"


def page(text: str, slugs: dict[str, str]) -> tuple[str, str]:
    """An ADR's markdown as a site page, and its title."""
    first, _, body = text.partition("\n")
    title = first.lstrip("# ").strip()
    body = re.sub(r"\]\((?:\.\./)+(results|docs|src|data)/", rf"]({REPO}/blob/main/\1/", body)
    body = re.sub(r"(?<![\w/\[`#])#(\d+)\b", rf"[#\1]({REPO}/issues/\1)", body)
    body = re.sub(
        r"\bADR (\d{4})\b(?!\]\()",
        lambda m: f"[ADR {m[1]}]({slugs[m[1]]}.html)" if m[1] in slugs else m[0],
        body,
    )
    body = re.sub(r"\]\((\d{4}-[\w-]+)\.md\)", r"](\1.html)", body)
    return f'---\ntitle: "{title.replace(chr(34), chr(39))}"\n---\n{body}', title


def main() -> None:
    assets = SITE / "assets"
    assets.mkdir(exist_ok=True)
    for f in [
        *(SITE.parent / "docs" / "figures").glob("*.json"),
        *(SITE.parent / "data" / "probe").glob("*.json"),
    ]:
        shutil.copy(f, assets / f.name)
    OUT.mkdir(exist_ok=True)
    files = sorted(ADRS.glob("[0-9][0-9][0-9][0-9]-*.md"))
    slugs = {f.name[:4]: f.stem for f in files}
    rows = []
    for f in files:
        text, title = page(f.read_text(), slugs)
        (OUT / f"{f.stem}.qmd").write_text(text)
        status = f.read_text().splitlines()[2]
        rows.append(f"| [{title}]({f.stem}.html) | {status} |")
    (OUT / "index.qmd").write_text(
        "---\ntitle: Decisions\n---\n\nEvery decision that shaped the experiments, with the "
        "options it weighed, as recorded in the repository's "
        f"[docs/adr]({REPO}/tree/main/docs/adr). Rendered here as written.\n\n"
        "| Decision | Date and status |\n|---|---|\n" + "\n".join(rows) + "\n"
    )


if __name__ == "__main__":
    main()
