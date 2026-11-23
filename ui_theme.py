"""
Gideon - Time Vault HUD theme (Level 3)

Dark obsidian background, cyan and electric-blue holographic glow, and an
animated Gideon core. No web fonts or external files are loaded: everything is
local, so the interface works fully offline. Motion is slow and restrained, and
it switches off completely if your system asks for reduced motion.
"""

from __future__ import annotations

import html

HUD_CSS = """
:root {
    --obsidian: #05080d;
    --panel: #09111a;
    --cyan: #3de9ff;
    --blue: #2f7bff;
    --text: #d7e6f2;
    --muted: #7f93a6;
    --amber: #ffb454;
    --red: #ff5d73;
}

.stApp {
    background: radial-gradient(1100px 520px at 50% -8%, #0d2236 0%, #05080d 62%);
    font-family: "Bahnschrift", "Segoe UI Variable Display", "Segoe UI", system-ui, sans-serif;
}
.block-container { max-width: 1500px; padding-top: 1.2rem; padding-bottom: 2rem; }
footer { visibility: hidden; }

/* Header */
.hud-title {
    font-size: 1.55rem; font-weight: 600; letter-spacing: 0.2em;
    color: var(--cyan); text-shadow: 0 0 14px rgba(61, 233, 255, 0.55);
}
.hud-sub { color: var(--muted); font-size: 0.85rem; letter-spacing: 0.04em; margin-bottom: 0.4rem; }

/* Holographic panels (containers created with key="panel_...") */
[class*="st-key-panel_"] {
    border: 1px solid rgba(61, 233, 255, 0.28);
    border-radius: 14px;
    padding: 1rem 1rem 0.6rem 1rem;
    background: linear-gradient(180deg, rgba(10, 22, 34, 0.86), rgba(7, 13, 21, 0.86));
    box-shadow: 0 0 0 1px rgba(47, 123, 255, 0.08) inset, 0 0 22px rgba(61, 233, 255, 0.10);
}
.panel-title { color: var(--cyan); font-weight: 600; letter-spacing: 0.06em; margin-bottom: 0.4rem; }
.panel-note { color: var(--muted); font-size: 0.85rem; }

/* Buttons and inputs */
.stButton > button {
    border: 1px solid rgba(61, 233, 255, 0.35);
    background: rgba(10, 22, 34, 0.7);
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
}
.stButton > button:hover {
    border-color: var(--cyan);
    box-shadow: 0 0 12px rgba(61, 233, 255, 0.35);
}
textarea:focus {
    box-shadow: 0 0 0 1px var(--cyan), 0 0 14px rgba(61, 233, 255, 0.3) !important;
}

/* Gideon's message */
.gideon-msg {
    font-size: 1.08rem; line-height: 1.55; color: var(--text);
    border-left: 2px solid var(--cyan); padding-left: 0.85rem; margin: 0.4rem 0 0.6rem 0;
}
.gideon-focus { font-size: 1.25rem; font-weight: 600; color: #e9fdff; margin: 0.2rem 0 0.6rem 0; }

/* Orb */
.orb-wrap { display: flex; flex-direction: column; align-items: center; margin: 0.2rem 0 0.8rem 0; }
.orb { position: relative; width: 150px; height: 150px; border-radius: 50%; }
.orb-core {
    position: absolute; inset: 26px; border-radius: 50%;
    background: radial-gradient(circle at 38% 32%, #e9fdff 0%, var(--orb-a) 28%, var(--orb-b) 62%, rgba(5, 8, 13, 0) 100%);
    box-shadow: 0 0 38px var(--orb-glow), 0 0 90px rgba(47, 123, 255, 0.35);
    animation: orb-breathe 5s ease-in-out infinite;
}
.orb-ring { position: absolute; inset: 0; border-radius: 50%; border: 1.5px solid transparent; }
.orb-ring.r1 {
    border-top-color: var(--orb-a); border-bottom-color: rgba(47, 123, 255, 0.4);
    box-shadow: 0 0 14px rgba(61, 233, 255, 0.25); animation: orb-spin 9s linear infinite;
}
.orb-ring.r2 {
    inset: 12px; border-left-color: var(--orb-b); border-right-color: rgba(61, 233, 255, 0.35);
    animation: orb-spin 14s linear infinite reverse;
}
.orb-idle, .orb-speaking, .orb-thinking { --orb-a: #3de9ff; --orb-b: #2f7bff; --orb-glow: rgba(61, 233, 255, 0.55); }
.orb-denied { --orb-a: #ff5d73; --orb-b: #c2243d; --orb-glow: rgba(255, 93, 115, 0.6); }
.orb-thinking .orb-core { animation-duration: 1.6s; }
.orb-thinking .orb-ring.r1 { animation-duration: 2.2s; }
.orb-thinking .orb-ring.r2 { animation-duration: 3.4s; }
.orb-label { margin-top: 0.5rem; color: var(--muted); font-size: 0.85rem; letter-spacing: 0.08em; }

@keyframes orb-breathe {
    0%, 100% { transform: scale(0.96); opacity: 0.9; }
    50% { transform: scale(1.04); opacity: 1; }
}
@keyframes orb-spin { to { transform: rotate(360deg); } }
@keyframes orb-speak {
    0%, 100% { transform: scale(0.95); }
    50% { transform: scale(1.14); }
}
@media (prefers-reduced-motion: reduce) {
    .orb-core, .orb-ring { animation: none !important; }
}

/* Focus Mode timer */
.gideon-timer {
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 4.5rem; font-weight: 600; letter-spacing: 0.02em;
    font-variant-numeric: tabular-nums; text-align: center;
    color: #e9fdff; text-shadow: 0 0 18px rgba(61, 233, 255, 0.5);
    margin: 0.5rem 0 0.75rem 0;
}

/* Reminders */
.rem {
    border-left: 3px solid var(--muted); padding: 0.35rem 0.6rem; margin: 0.1rem 0 0.4rem 0;
    border-radius: 6px; background: rgba(61, 233, 255, 0.04);
}
.rem-badge { font-size: 0.78rem; font-weight: 600; margin-right: 0.4rem; }
.rem-date { font-size: 0.78rem; color: var(--muted); }
.urg-overdue { border-left-color: var(--red); }
.urg-overdue .rem-badge { color: var(--red); }
.urg-today { border-left-color: var(--amber); }
.urg-today .rem-badge { color: var(--amber); }
.urg-soon { border-left-color: var(--cyan); }
.urg-soon .rem-badge { color: var(--cyan); }
.urg-later { border-left-color: var(--blue); }
.urg-later .rem-badge { color: var(--blue); }
"""


def orb_html(state: str = "idle", label: str = "Standing by", speak_seconds: float = 0.0) -> str:
    """HTML for the Gideon core. state: idle, thinking, speaking, or denied."""
    core_style = ""
    if state == "speaking" and speak_seconds > 0:
        pulses = max(1, round(speak_seconds / 0.9))
        core_style = (
            f' style="animation: orb-speak 0.9s ease-in-out {pulses}, '
            f'orb-breathe 5s ease-in-out {pulses * 0.9:.1f}s infinite;"'
        )
    return (
        '<div class="orb-wrap">'
        f'<div class="orb orb-{html.escape(state)}">'
        '<div class="orb-ring r1"></div><div class="orb-ring r2"></div>'
        f'<div class="orb-core"{core_style}></div>'
        "</div>"
        f'<div class="orb-label">{html.escape(label)}</div>'
        "</div>"
    )
