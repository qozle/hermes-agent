"""Post kanban card events to Max's dashboard notification centre.

WHY THIS EXISTS. Card t_325f9028 consolidates every alert and FYI on the stack
onto one destination. Two kanban events were still arriving as WhatsApp pings:

  * a card entering ``blocked`` with kind ``needs_input`` -- RED, because that
    is the board asking Max a question and nothing moves until he answers;
  * a card reaching ``done`` -- YELLOW, so he learns when work lands.

Those two are CLAIMED here: posted to the centre, and the platform ping is
skipped so he is not told twice. Every other event kind (``crashed``,
``timed_out``, ``review_requested``, a non-``needs_input`` block, ...) is left
alone and still goes out over the subscription's platform -- narrowing the
blast radius to exactly what the card asked for.

THIS NEVER RAISES INTO THE NOTIFIER. A claim that fails to post returns False,
which means "not claimed", which means the normal platform ping happens after
all. A dashboard outage therefore degrades to the old behaviour instead of
losing Max's notification -- the one failure mode that would actually cost him
something.

CONFIG. Off unless a destination is configured. ``NOTIFY_URL`` /
``NOTIFY_TOKEN`` from the environment, else ``~/.hermes/notify.env``
(``$NOTIFY_ENV`` overrides the file). The token is the poster's credential --
a gateway process reaching the dashboard sends no Tailscale-User-Login header,
so the identity check alone would refuse it. It is never logged and never put
on argv.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Serve terminates TLS for the dashboard on the tailnet; :8081 is loopback-only.
DEFAULT_URL = "https://datamachine.tail3f008b.ts.net:8443"
API_PATH = "/dashboard/api/notifications"
TIMEOUT_S = 10
SOURCE = "kanban"

#: How many characters of a reason/handoff survive into the body. Generous
#: compared with a phone ping because the centre has room to scroll, but still
#: bounded: the API stores what it is given.
BODY_CLIP = 1200


def _env_file() -> Path:
    """Where the notify credentials live.

    Honour ``HERMES_HOME`` rather than hard-coding ``~/.hermes``: a gateway
    running under a named profile has its own home, and a test harness points
    it at a tmpdir. Reading the real ``~/.hermes`` regardless would make this
    module read one box's credentials from another box's process -- and the
    test suite rightly fails any file I/O against the real home.
    """
    explicit = os.environ.get("NOTIFY_ENV")
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("HERMES_HOME")
    base = Path(home).expanduser() if home else (Path.home() / ".hermes")
    return base / "notify.env"


def _dotenv(keys: tuple[str, ...]) -> dict[str, str]:
    """Read named keys from the notify env file. Values stay in memory."""
    out: dict[str, str] = {}
    try:
        text = _env_file().read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        if k in keys and k not in out:
            out[k] = v.strip().strip('"').strip("'")
    return out


def resolve_token() -> str:
    tok = (os.environ.get("NOTIFY_TOKEN") or "").strip()
    return tok or _dotenv(("NOTIFY_TOKEN",)).get("NOTIFY_TOKEN", "")


def resolve_base_url() -> str:
    url = (os.environ.get("NOTIFY_URL") or "").strip()
    return (url or _dotenv(("NOTIFY_URL",)).get("NOTIFY_URL", "") or DEFAULT_URL).rstrip("/")


def enabled() -> bool:
    """A token is the switch: no token means nothing was ever configured here."""
    if (os.environ.get("KANBAN_NOTIFY_CENTRE") or "").strip().lower() in {"0", "off", "false"}:
        return False
    return bool(resolve_token())


def _clip(text: str, limit: int = BODY_CLIP) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def classify(ev: Any) -> tuple[str, str] | None:
    """(level, headline prefix) for the two kinds this module claims, else None.

    Keep this total and boring: anything not explicitly named here must fall
    through to the platform ping, because an event silently swallowed by a
    misfiring classifier is an event Max never sees anywhere.
    """
    kind = getattr(ev, "kind", None)
    payload = getattr(ev, "payload", None) or {}
    if kind == "completed":
        return "yellow", "Done"
    if kind == "blocked" and payload.get("kind") == "needs_input":
        return "red", "Needs you"
    return None


def _artifact_lines(payload: dict) -> list[str]:
    """Attachment/artifact names from a completion handoff, if any.

    Names, not paths: the board stores them under
    ``~/.hermes/kanban/attachments/<task>/`` on the host, which is not a URL
    and must not be passed off as a tappable link.
    """
    arts = payload.get("artifacts") or (payload.get("metadata") or {}).get("artifacts") or []
    if isinstance(arts, str):
        arts = [arts]
    out = []
    for a in arts:
        if isinstance(a, str) and a.strip():
            out.append(Path(a).name)
    return out


def build(ev: Any, task: Any, task_id: str) -> tuple[str, str, str] | None:
    """(level, title, body) for a claimable event, else None."""
    verdict = classify(ev)
    if verdict is None:
        return None
    level, prefix = verdict
    payload = getattr(ev, "payload", None) or {}
    title_text = (getattr(task, "title", None) or task_id or "").strip() or task_id
    title = f"{prefix}: {title_text}"

    parts: list[str] = []
    if level == "red":
        reason = payload.get("reason") or payload.get("note") or ""
        parts.append(_clip(reason) if reason else "The card is blocked on your input.")
    else:
        summary = payload.get("summary") or payload.get("result") or ""
        if summary:
            parts.append(_clip(summary))
        names = _artifact_lines(payload)
        if names:
            parts.append("Attached: " + ", ".join(names[:8]))
    parts.append(f"Card {task_id}")
    return level, title, "\n\n".join(p for p in parts if p)


def post(level: str, source: str, title: str, body: str = "",
         link: str | None = None, base_url: str | None = None,
         token: str | None = None) -> dict:
    """POST one notification. Raises on failure; ``claim`` swallows."""
    base = (base_url or resolve_base_url()).rstrip("/")
    data = json.dumps({
        "level": level, "source": source, "title": title,
        "body": body or "", "link": link or None,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base}{API_PATH}", data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    tok = resolve_token() if token is None else token
    if tok:
        req.add_header("X-Notify-Token", tok)
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


def claim(ev: Any, task: Any, task_id: str) -> bool:
    """Post this event to the centre and report whether the platform ping is now redundant.

    False always means "carry on as before". That covers an event this module
    does not handle, a centre that was never configured, AND a post that
    failed -- in the last case the platform ping is the fallback, which is
    exactly what should happen when the new destination is unreachable.

    Blocking I/O: call from a worker thread, not the event loop.
    """
    if not enabled():
        return False
    built = build(ev, task, task_id)
    if built is None:
        return False
    level, title, body = built
    try:
        post(level, SOURCE, title, body)
    except urllib.error.HTTPError as exc:
        logger.warning("kanban notify centre: dashboard refused %s for %s: HTTP %s",
                       getattr(ev, "kind", "?"), task_id, exc.code)
        return False
    except Exception as exc:  # URLError, timeout, DNS, bad JSON -- anything
        logger.warning("kanban notify centre: could not post %s for %s (%s: %s)",
                       getattr(ev, "kind", "?"), task_id, type(exc).__name__, exc)
        return False
    logger.debug("kanban notify centre: posted %s %s for %s", level, getattr(ev, "kind", "?"), task_id)
    return True
