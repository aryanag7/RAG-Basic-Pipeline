from typing import Literal

Stage = Literal["retrieve", "generate", "complete"]

_NODES = [
    ("question", "📝", "Question"),
    ("retrieve", "🔎", "Search + fuse"),
    ("generate", "🤖", "Generate answer"),
    ("answer", "💡", "Answer"),
]

_STAGE_STATES: dict[Stage, dict[str, str]] = {
    "retrieve": {"question": "done", "retrieve": "active", "generate": "idle", "answer": "idle"},
    "generate": {"question": "done", "retrieve": "done", "generate": "active", "answer": "idle"},
    "complete": {"question": "done", "retrieve": "done", "generate": "done", "answer": "done"},
}

_FLOWING_CONNECTOR: dict[Stage, int] = {"retrieve": 0, "generate": 1, "complete": 2}

_STATUS_BADGES = {
    "added": ":green-badge[Indexed]",
    "duplicate": ":gray-badge[Already indexed]",
    "empty": ":orange-badge[No text found]",
    "error": ":red-badge[Failed]",
}


def format_file_size(num_bytes: int) -> str:
    """Format a byte count as a human-readable size, e.g. 1536000 -> '1.5 MB'."""

    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def status_badge(status: str) -> str:
    """Map a processed_uploads status to a colored markdown badge shortcode."""

    return _STATUS_BADGES.get(status, ":gray-badge[Unknown]")


def render_pipeline_frame(stage: Stage) -> str:
    """Render one frame of the retrieval/generation pipeline diagram."""

    states = _STAGE_STATES[stage]
    flowing_index = _FLOWING_CONNECTOR[stage]
    parts = ['<div class="pl-row">']

    for index, (key, icon, label) in enumerate(_NODES):
        state = states[key]
        sub = ""
        if key == "retrieve":
            sub = (
                '<div class="pl-sub">'
                '<span class="pl-chip">Embeddings</span>'
                '<span class="pl-chip-op">+</span>'
                '<span class="pl-chip">BM25</span>'
                '<div class="pl-fuse">&#8594; RRF fusion</div>'
                "</div>"
            )
        parts.append(
            f'<div class="pl-node pl-{state}">'
            f'<div class="pl-icon">{icon}</div>'
            f'<div class="pl-label">{label}</div>{sub}</div>'
        )
        if index < len(_NODES) - 1:
            flowing = "pl-flowing" if index == flowing_index else ""
            parts.append(f'<div class="pl-connector {flowing}"></div>')

    parts.append("</div>")
    return "".join(parts)


CUSTOM_CSS = """
<style>
.pl-row {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0.25rem;
    padding: 1.25rem 0.5rem;
}

.pl-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.15rem;
    min-width: 108px;
    padding: 0.75rem 0.5rem;
    border-radius: 12px;
    border: 1.5px solid #E8E1D3;
    background: #F3ECDE;
    text-align: center;
    transition: all 0.25s ease;
}

.pl-icon { font-size: 1.6rem; line-height: 1; }
.pl-label { font-size: 0.8rem; font-weight: 600; color: #1F2430; }

.pl-idle { opacity: 0.5; }

.pl-active {
    border-color: #E4572E;
    background: #FCEFE8;
    animation: pl-pulse 1.3s ease-in-out infinite;
}
.pl-active .pl-icon { animation: pl-bounce 1.3s ease-in-out infinite; }

.pl-done { border-color: #2F6F5E; background: #EAF2EF; }
.pl-done .pl-label::after { content: " \\2713"; color: #2F6F5E; }

.pl-sub {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.2rem;
    margin-top: 0.2rem;
}
.pl-chip {
    font-size: 0.65rem;
    padding: 0.05rem 0.4rem;
    border-radius: 999px;
    background: #FFFFFF;
    border: 1px solid #E8E1D3;
    color: #1F2430;
}
.pl-chip-op { font-size: 0.65rem; opacity: 0.6; }
.pl-fuse { font-size: 0.65rem; opacity: 0.7; margin-top: 0.1rem; }

.pl-connector {
    flex: 1;
    min-width: 24px;
    height: 2px;
    background-image: repeating-linear-gradient(90deg, #E4572E 0 6px, transparent 6px 12px);
    background-size: 12px 2px;
    opacity: 0.35;
}
.pl-connector.pl-flowing {
    opacity: 1;
    animation: pl-flow 0.6s linear infinite;
}

@keyframes pl-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(228, 87, 46, 0.35); }
    50% { box-shadow: 0 0 0 7px rgba(228, 87, 46, 0); }
}
@keyframes pl-bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-3px); }
}
@keyframes pl-flow {
    from { background-position: 0 0; }
    to { background-position: 12px 0; }
}

@media (max-width: 640px) {
    .pl-connector { min-width: 12px; }
    .pl-node { min-width: 84px; padding: 0.5rem 0.35rem; }
}

.st-key-answer_card {
    border-left: 4px solid #E4572E !important;
    background: #FBF7F0;
}
</style>
"""
