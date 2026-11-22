"""A tiny key-value-store agent, used as an example agent under test.

The agent is stateful on purpose: it keeps one in-memory dict, so a suite
can SET a key in one case and GET it in a later one. Commands are plain
text, one per prompt:

    SET <key> <value>   -> "OK"
    GET <key>           -> the value, or "nil"
    DEL <key>           -> "1" if the key existed, else "0"
    EXISTS <key>        -> "1" if the key exists, else "0"

Use :func:`make_agent` for a fresh store; the module-level ``agent`` is one
shared instance, convenient for the CLI.
"""

from __future__ import annotations

from typing import Callable


def make_agent() -> Callable[[str], str]:
    store: dict = {}

    def agent(prompt: str) -> str:
        parts = prompt.strip().split(None, 2)
        if not parts:
            return "ERR empty command"
        cmd = parts[0].upper()
        if cmd == "SET" and len(parts) == 3:
            store[parts[1]] = parts[2]
            return "OK"
        if cmd == "GET" and len(parts) == 2:
            return store.get(parts[1], "nil")
        if cmd == "DEL" and len(parts) == 2:
            return "1" if store.pop(parts[1], None) is not None else "0"
        if cmd == "EXISTS" and len(parts) == 2:
            return "1" if parts[1] in store else "0"
        return "ERR unknown command"

    return agent


agent = make_agent()
