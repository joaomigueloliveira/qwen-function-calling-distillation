"""Pre-render step, generating what the site needs from the repo (neither is committed):
- docs/adr/*.md become the Decisions pages (decisions/);
- the figures' data (docs/figures/*.json) and the probes (data/probe/*.json) go to assets/;
- assets/og-card.png, the link preview (title and the headline ladder), is drawn from
  results/methods.json.

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


def og_card(out: Path) -> None:
    """The link preview: 1200 x 630, the title and the four test-clean scores on one track."""
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    from fc_distill.story import HEADLINE, Story

    story = Story.load()
    points = [
        ("raw 0.8B", "the 0.8B before", "#898781", -1),
        ("raw 2B", "its 2B teacher", "#52514e", 1),
        (HEADLINE, "the 0.8B, distilled\nwithout labels", "#2a78d6", -1),
        ("A", "trained on\ngold labels", "#52514e", 1),
    ]
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor="#fdfdfc")
    fig.text(
        0.06,
        0.82,
        "Distilling function calling without labels",
        fontsize=34,
        family="serif",
        color="#1b1b1a",
    )
    fig.text(
        0.06,
        0.73,
        "Qwen3.5-2B teaches Qwen3.5-0.8B to call tools, on one 12 GB GPU",
        fontsize=17,
        color="#52514e",
    )
    ax = fig.add_axes((0.08, 0.12, 0.84, 0.45))
    ax.set_xlim(50, 100)
    ax.set_ylim(-1.6, 1.6)
    ax.axis("off")
    ax.plot([50, 100], [0, 0], color="#e6e4dd", lw=3, zorder=1)
    for t in range(50, 101, 10):
        ax.text(t, -0.28, f"{t}%", ha="center", va="top", fontsize=12, color="#898781")
    for who, label, colour, side in points:
        x = float(story.score(who).rstrip("%"))
        hollow = who == "A"
        ax.scatter(
            [x],
            [0],
            s=260,
            color="#fdfdfc" if hollow else colour,
            edgecolor=colour,
            linewidth=2.5,
            zorder=3,
        )
        y = 0.5 if side > 0 else -0.62
        va = "bottom" if side > 0 else "top"
        ax.text(x, y, f"{x:.1f}%", ha="center", va=va, fontsize=24, weight="bold", color=colour)
        ax.text(
            x,
            y + (0.42 if side > 0 else -0.42),
            label,
            ha="center",
            va=va,
            fontsize=13,
            color="#52514e",
        )
    fig.text(0.06, 0.05, "Exact match on xLAM test-clean", fontsize=12, color="#898781")
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    assets = SITE / "assets"
    assets.mkdir(exist_ok=True)
    og_card(assets / "og-card.png")
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
        line = f.read_text().splitlines()[2]
        date = re.search(r"Date: ([\d-]+)", line)
        state = re.search(r"Status: (\w+)", line)
        amended = re.findall(r"(\d{4}-\d{2}-\d{2})", line.split("Status:", 1)[-1])
        badge = f'<span class="status">{state[1] if state else "?"}</span>'
        when = f'<span class="amended">amended {max(amended)}</span>' if amended else ""
        rows.append(f"| [{title}]({f.stem}.html) | {date[1] if date else ''} | {badge} {when} |")
    (OUT / "index.qmd").write_text(
        "---\ntitle: Decisions\n---\n\nEvery decision that shaped the experiments, with the "
        "options it weighed, as recorded in the repository's "
        f"[docs/adr]({REPO}/tree/main/docs/adr). Rendered here as written.\n\n"
        "| Decision | Date | Status |\n|---|---|---|\n" + "\n".join(rows) + "\n"
    )


if __name__ == "__main__":
    main()
