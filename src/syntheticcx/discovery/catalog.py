"""Private, revision-scoped locator catalogs compiled only from observed metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from syntheticcx.config.models import LocatorConfig


def css_string(value: str) -> str:
    """CSS quoted-string escaping (including control characters, not identifier escaping)."""
    return (
        '"'
        + "".join(
            "\\" + char
            if char in {'"', "\\"}
            else f"\\{ord(char):x} "
            if ord(char) < 32 or ord(char) == 127
            else char
            for char in value
        )
        + '"'
    )


def node_locators(node: dict[str, Any]) -> list[LocatorConfig]:
    """Reusable semantic/attribute candidates; never interpolate prose or raw classes."""
    tag = str(node.get("tag", ""))
    if not re.fullmatch(r"[a-z][a-z0-9-]*", tag):
        return []
    attrs = node.get("attributes", {})
    if not isinstance(attrs, dict):
        return []
    result: list[LocatorConfig] = []
    if isinstance(attrs.get("data-testid"), str) and attrs["data-testid"]:
        result.append(LocatorConfig(strategy="test_id", value=attrs["data-testid"]))
    if node.get("editable") and attrs.get("aria-label"):
        result.append(LocatorConfig(strategy="label", value=attrs["aria-label"]))
    if node.get("editable") and attrs.get("placeholder"):
        result.append(LocatorConfig(strategy="placeholder", value=attrs["placeholder"]))
    if attrs.get("role") and attrs.get("aria-label"):
        result.append(LocatorConfig(strategy="role", value=attrs["role"], name=attrs["aria-label"]))
    for name in ("data-actor", "data-side", "data-role", "role", "aria-label", "id"):
        value = attrs.get(name)
        if isinstance(value, str) and value and len(value) < 160:
            result.append(LocatorConfig(strategy="css", value=f"{tag}[{name}={css_string(value)}]"))
    classes = str(attrs.get("class", "")).split()[:8]
    if classes:
        result.append(
            LocatorConfig(
                strategy="css",
                value=tag + "".join(f"[class~={css_string(value)}]" for value in classes),
            )
        )
        result.extend(
            LocatorConfig(strategy="css", value=f"{tag}[class~={css_string(value)}]")
            for value in classes
        )
    # A tag alone is observed, but still requires bounded native ownership validation.
    result.append(LocatorConfig(strategy="css", value=tag))
    return list({item.model_dump_json(): item for item in result}.values())


@dataclass
class CatalogItem:
    kind: str
    value: Any
    description: str


@dataclass
class LocatorCatalog:
    revision: str = field(default_factory=lambda: uuid4().hex[:16])
    items: dict[str, CatalogItem] = field(default_factory=dict)

    def add(self, kind: str, value: Any, description: str) -> str | None:
        siblings = [item for item in self.items.values() if item.kind == kind]
        if any(item.value == value for item in siblings):
            return None
        if len(siblings) >= 12 or len(self.items) >= 64:
            return None
        key = f"{kind}-{len(siblings) + 1}"
        self.items[key] = CatalogItem(kind, value, description[:240])
        return key

    def resolve(self, revision: str, key: str, kind: str) -> Any:
        if revision != self.revision:
            raise ValueError("unknown catalog revision")
        item = self.items.get(key)
        if item is None or item.kind != kind:
            raise ValueError("unknown or wrong-kind catalog candidate")
        return item.value

    def public(self) -> dict[str, Any]:
        """Bounded descriptions only; callers redact observed metadata before model use."""
        return {
            "revision": self.revision,
            "candidates": [
                {"id": key, "kind": item.kind, "description": item.description}
                for key, item in self.items.items()
            ],
        }

    def add_node(self, kind: str, node: dict[str, Any]) -> None:
        for index, locator in enumerate(node_locators(node)):
            self.add(
                kind,
                locator,
                f"Observed {node.get('tag', 'element')} "
                f"{locator.strategy} structure candidate {index + 1}; "
                f"dialog={bool(node.get('in_dialog'))}; "
                f"metadata {str(node.get('attributes', {}))[:120]}",
            )


def interaction_catalog(
    snapshot: str, inspection: str, chat_hint: str | None = None
) -> LocatorCatalog:
    """Snapshot refs are action evidence; inventory supplies reusable interaction locators."""
    import json

    from syntheticcx.adapters.discovery.playwright_mcp import _CHAT_WORDS, _UNSAFE_WORDS

    catalog = LocatorCatalog()
    for match in re.finditer(
        r'^\s*-\s*(button|link)\s+"([^"\n]{1,256})"[^\n]*\[ref=(e\d+)\]', snapshot, re.MULTILINE
    ):
        label = " ".join(match[2].casefold().split())
        if _UNSAFE_WORDS.search(label):
            continue
        if not _CHAT_WORDS.search(label) and not (
            chat_hint and " ".join(chat_hint.casefold().split()) in label
        ):
            continue
        catalog.add("opener", (match[3], match[2]), f"Current {match[1]}: {match[2]}")
    try:
        data = json.loads(inspection)
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    inventories = [
        data,
        *[
            frame
            for frame in data.get("frames", [])
            if isinstance(frame, dict) and not frame.get("inaccessible")
        ],
    ]
    dialog_editors = any(
        node.get("visible")
        and node.get("editable")
        and (node.get("in_dialog") or inventory.get("in_dialog"))
        for inventory in inventories
        for node in inventory.get("elements", [])
        if isinstance(node, dict)
    )
    # A page-level search/newsletter field cannot replace an observed dialog composer.
    active_dialog_indent: int | None = None
    for line in snapshot.splitlines():
        indent = len(line) - len(line.lstrip())
        if re.match(r"\s*-\s*dialog\b", line):
            active_dialog_indent = indent
        elif active_dialog_indent is not None and line.strip() and indent <= active_dialog_indent:
            active_dialog_indent = None
        button_match = re.match(r'\s*-\s*button\s+"([^"\n]{1,256})"', line)
        if (
            button_match
            and re.fullmatch(r"send|enviar", button_match[1], re.I)
            and (not dialog_editors or active_dialog_indent is not None)
        ):
            catalog.add(
                "send",
                LocatorConfig(strategy="role", value="button", name=button_match[1]),
                f"Observed send button: {button_match[1]}",
            )
    for inventory in inventories:
        if isinstance(inventory.get("frame"), dict):
            catalog.add_node("frame", inventory["frame"])
        for node in inventory.get("elements", []):
            if not isinstance(node, dict) or not node.get("visible"):
                continue
            in_dialog = node.get("in_dialog") or inventory.get("in_dialog")
            if node.get("editable") and (not dialog_editors or in_dialog):
                catalog.add_node("input", node)
            attrs = node.get("attributes", {})
            label = " ".join(str(attrs.get("aria-label", "")).casefold().split())
            labeled_send = bool(
                re.fullmatch(r"send(?: (?:a )?message)?|enviar(?: (?:uma )?mensagem)?", label)
            )
            # An observed, explicitly named send ID in the opened dialog is evidence,
            # not a website-specific selector or permission to select arbitrary buttons.
            identifier = str(attrs.get("id", "")).casefold()
            identified_send = (
                (node.get("in_dialog") or inventory.get("in_dialog"))
                and node.get("tag") == "button"
                and attrs.get("type") != "reset"
                and not label
                and re.search(r"(?:^|[-_:])(?:send|enviar)(?:$|[-_:])", identifier)
                and not re.search(
                    r"purchase|payment|buy|checkout|upload|attach|login|consent|"
                    r"pagar|comprar|anexar|senha|aceitar",
                    identifier,
                )
            )
            if (labeled_send or identified_send) and (not dialog_editors or in_dialog):
                catalog.add_node("send", node)
    # Prefer the inventory's current-dialog editors before snapshot-only fallbacks.
    for match in re.finditer(r'^\s*-\s*textbox\s+"([^"\n]{1,256})"', snapshot, re.MULTILINE):
        if dialog_editors:
            break
        catalog.add(
            "input",
            LocatorConfig(strategy="role", value="textbox", name=match[1]),
            f"Visible editable textbox: {match[1]}",
        )
    return catalog
