"""Native exchange evidence for unmarked, structurally separated message readers."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from playwright.async_api import ElementHandle

from syntheticcx.adapters.channel.ownership import OwnedScope
from syntheticcx.adapters.channel.playwright import locate, normalize_echo
from syntheticcx.adapters.channel.reading import READING_HELPERS
from syntheticcx.config.models import LocatorConfig
from syntheticcx.errors import ChannelError

ReaderCandidate = tuple[Literal["assistant", "customer"], LocatorConfig, ElementHandle, str]

_HELPERS = (
    READING_HELPERS
    + """
const forbiddenReader = 'html,body,form,button,a,input,textarea,select,dialog,nav,main,' +
  'header,footer,h1,h2,h3,h4,h5,h6,time,script,style,template,noscript,' +
  '[role=dialog],[role=log],[role=feed],[role=status],[role=alert],' +
  '[role=button],[role=textbox],[contenteditable]:not([contenteditable=false])';
const excludedAncestor = 'header,footer,nav,h1,h2,h3,h4,h5,h6,[role=status],[role=alert],' +
  'input,textarea,select,[role=textbox],[contenteditable]:not([contenteditable=false])';
const safeLeaf = (root, node) => {
  if (node === root || !node.isConnected || !composedContains(root, node) ||
      node.matches(forbiddenReader) || !renderedElement(node)) return false;
  const scan = walkDOM(node);
  if (!scan.complete || scan.nodes.some(child => child !== node &&
      child.matches('input,textarea,select,[role=textbox],' +
        '[contenteditable]:not([contenteditable=false])'))) return false;
  for (let parent = node; parent && parent !== root;
       parent = parent.parentElement || parent.getRootNode().host) {
    if (parent.matches(excludedAncestor) || observedActorConflict(parent)) return false;
  }
  // An inline fragment cannot discard surrounding message prose. Whole inline
  // messages remain eligible; timestamps and controls are supplemental evidence.
  const inline = 'span,strong,em,b,i,u,small,code';
  if (node.matches(inline)) {
    let child = node;
    for (let parent = composedParent(child); parent && parent !== root;
         child = parent, parent = composedParent(parent)) {
      const siblings = [...parent.childNodes].filter(part => part !== child);
      if (siblings.some(part => part.nodeType === 3 ? !!part.textContent.trim() :
        part.nodeType === 1 && !part.matches(controls + ',time') && !!proseText(part)))
        return false;
      if (!parent.matches(inline)) break;
    }
  }
  return !!proseText(node);
};
const overlap = (a, b) => a === b || composedContains(a, b) || composedContains(b, a);
const readerActor = (root, node) => {
  for (let current = node; current && current !== root; current = composedParent(current)) {
    const actor = observedActor(current);
    if (actor) return actor;
  }
  return null;
};
const customerMessage = (root, echo, before) => {
  let branch = echo;
  for (let parent = composedParent(branch); parent; branch = parent,
       parent = composedParent(parent)) {
    // Some widgets append one exchange row containing separate customer and
    // assistant branches. Do not claim their common fresh row as the customer.
    if (branch.matches('div,article,section,li') &&
        [...parent.children].some(sibling => sibling !== branch &&
          !before.includes(sibling) && !sibling.matches('time,[role=tooltip]') &&
          safeLeaf(root, sibling) && !!proseText(sibling))) return branch;
    if (parent === root || before.includes(parent)) {
      // A fresh message container is needed to separate its ancillary metadata
      // from an independently rendered response. Flat/reused nodes are ambiguous.
      return branch.matches('div,article,section,li') && !before.includes(branch) ? branch : root;
    }
    if (!composedContains(root, parent)) break;
  }
  return root;
};
const structuralSelectors = (root, node) => {
  const part = el => el.tagName.toLowerCase() + [...el.classList].slice(0, 8)
    .map(value => '.' + CSS.escape(value)).join('');
  const result = [part(node)];
  let child = node;
  let selector = result[0];
  for (let depth = 0; depth < 3; depth++) {
    const parent = child.parentElement;
    if (!parent || parent === root || !composedContains(root, parent)) break;
    selector = part(parent) + ' > ' + selector;
    if (selector.length <= 1024) result.push(selector);
    child = parent;
  }
  return result;
};
"""
)


class ExchangeReaderEvidence:
    """Read-only corroboration tied to one actual submission and an attested scope.

    The caller invokes begin before dispatch, then exposes only compiled candidate
    IDs to its controller. No model-authored locator enters this helper's catalog.
    Returned handles remain owned by this object until begin/close is called.
    """

    def __init__(self, owned: OwnedScope) -> None:
        self.owned = owned
        self._before: list[ElementHandle] = []
        self._current: list[ElementHandle] = []
        self._text: str | None = None

    async def close(self) -> None:
        handles, self._before, self._current = self._before + self._current, [], []
        self._text = None
        await asyncio.gather(*(handle.dispose() for handle in handles), return_exceptions=True)

    async def begin(self, text: str) -> None:
        await self.owned.check()
        if not normalize_echo(text):
            raise ChannelError("channel.reader", "exchange requires nonempty submitted text")
        await self.close()
        self._before = await self.owned.locator.locator("*").element_handles()
        if len(self._before) > 2000:
            await self.close()
            raise ChannelError("channel.reader", "exchange node inventory exceeds limit")
        self._text = text

    async def _inventory(self) -> tuple[list[ElementHandle], list[dict[str, Any]]]:
        await self.owned.check()
        if self._text is None:
            raise ChannelError("channel.reader", "exchange baseline is missing")
        # Keep previously returned handles alive while a broker catalog refers to them.
        nodes = await self.owned.locator.locator("*").element_handles()
        if len(nodes) > 2000:
            for node in nodes:
                await node.dispose()
            raise ChannelError("channel.reader", "exchange node inventory exceeds limit")
        try:
            metadata = await self.owned.root.evaluate(
                "(root, args) => {"
                + _HELPERS
                + """
            return args.nodes.map((node, index) => {
              const safe = safeLeaf(root, node);
              return {index, known: args.known.indexOf(node),
                fresh: !args.before.includes(node), safe, actor: readerActor(root, node),
                control: node.matches('button,[role=button]') && renderedElement(node),
                text: safe ? proseText(node) : '',
                selectors: safe ? structuralSelectors(root, node) : []};
            });
            }""",
                {"nodes": nodes, "before": self._before, "known": self._current},
            )
        except BaseException:
            await asyncio.gather(*(node.dispose() for node in nodes), return_exceptions=True)
            raise
        retained = []
        for node, item in zip(nodes, metadata, strict=True):
            if item["known"] >= 0:
                retained.append(self._current[item["known"]])
                await node.dispose()
            else:
                self._current.append(node)
                retained.append(node)
        if len(self._current) > 20_000:
            raise ChannelError("channel.reader", "exchange retained node inventory exceeds limit")
        return retained, metadata

    async def _fresh_echoes(
        self, nodes: list[ElementHandle], metadata: list[dict[str, Any]], text: str
    ) -> list[ElementHandle]:
        echoes = [
            nodes[item["index"]]
            for item in metadata
            if item["fresh"]
            and item["safe"]
            and item["actor"] != "assistant"
            and normalize_echo(item["text"]) == normalize_echo(text)
        ]
        indices = await self.owned.root.evaluate(
            "(root, echoes) => {"
            + _HELPERS
            + """
            return echoes.map((node, index) => ({node, index})).filter(item =>
              !echoes.some(other => other !== item.node &&
                composedContains(item.node, other))).map(item => item.index);
            }""",
            echoes,
        )
        return [echoes[index] for index in indices]

    async def _fresh_controls(
        self,
        nodes: list[ElementHandle],
        metadata: list[dict[str, Any]],
        echoes: list[ElementHandle],
    ) -> list[ElementHandle]:
        """Visible controls appended by this exchange, outside the customer container."""
        fresh = [nodes[item["index"]] for item in metadata if item["fresh"] and item.get("control")]
        if not fresh:
            return []
        indices = await self.owned.root.evaluate(
            "(root, args) => {"
            + _HELPERS
            + """
            return args.controls.map((node, index) => ({node, index})).filter(item =>
              !args.echoes.some(echo => overlap(item.node,
                customerMessage(args.root, echo, args.before)))).map(item => item.index);
            }""",
            {"controls": fresh, "echoes": echoes, "root": self.owned.root, "before": self._before},
        )
        return [fresh[int(index)] for index in indices]

    async def candidates(self) -> list[ReaderCandidate]:
        nodes, metadata = await self._inventory()
        assert self._text is not None
        echoes = await self._fresh_echoes(nodes, metadata, self._text)
        if not echoes:
            return []
        if len(echoes) != 1:
            raise ChannelError("channel.reader", "fresh submitted-text echo is absent or ambiguous")
        controls = await self._fresh_controls(nodes, metadata, echoes)
        result: list[ReaderCandidate] = []
        # Assistant pool entries carry how many exchange controls their readers own;
        # a reply container that owns its menu is the only reader that can later
        # offer those controls as bound-state options.
        assistant_pool: list[tuple[ReaderCandidate, int]] = []
        seen: set[tuple[str, str]] = set()
        for item in metadata:
            if not item["fresh"] or not item["safe"]:
                continue
            node = nodes[int(item["index"])]
            customer = normalize_echo(str(item["text"])) == normalize_echo(self._text)
            if not customer and item["actor"] == "customer":
                continue
            if not customer and await node.evaluate(
                "(node, args) => {"
                + _HELPERS
                + "return args.echoes.some(echo => overlap(node, "
                + "customerMessage(args.root, echo, args.before)));}",
                {"echoes": echoes, "root": self.owned.root, "before": self._before},
            ):
                continue
            kind: Literal["assistant", "customer"] = "customer" if customer else "assistant"
            capped = (
                sum(entry[0] == "customer" for entry in result) >= 12
                if customer
                else len(assistant_pool) >= 24
            )
            if capped:
                continue
            for selector in item["selectors"]:
                config = LocatorConfig(strategy="css", value=str(selector))
                key = kind, config.value
                if key in seen:
                    continue
                safe, owned = await self._candidate_safe(
                    config, echoes, customer=customer, controls=[] if customer else controls
                )
                if not safe:
                    continue
                seen.add(key)
                description = (
                    "Fresh exact submitted-text echo" if customer else str(item["text"])[:240]
                )
                candidate: ReaderCandidate = (kind, config, node, description)
                if customer:
                    if sum(entry[0] == "customer" for entry in result) >= 12:
                        continue
                    result.append(candidate)
                else:
                    if len(assistant_pool) >= 24:
                        continue
                    assistant_pool.append((candidate, owned))
        if assistant_pool:
            # Prefer complete witnessed message branches over nested fragments
            # (including legal/AI footers). The planner must not select a footer
            # as the entire response merely because it has a convenient tag.
            complete = await self.owned.root.evaluate(
                "(root, nodes) => {"
                + _HELPERS
                + "return nodes.map((node, index) => index).filter(index => "
                "!nodes.some(other => other !== nodes[index] && "
                "composedContains(other, nodes[index]) && "
                "proseText(other) !== proseText(nodes[index])));}",
                [candidate[2] for candidate, _ in assistant_pool],
            )
            assistant_pool = [assistant_pool[index] for index in complete]
            owners = [(candidate, owned) for candidate, owned in assistant_pool if owned > 0]
            chosen = (
                owners[:12] if owners else [(candidate, 0) for candidate, _ in assistant_pool[:12]]
            )
            for candidate, owned in chosen:
                kind, config, node, description = candidate
                if owned:
                    description = f"{description}; owns {owned} fresh reply controls"
                result.append((kind, config, node, description[:240]))
        return result

    async def _candidate_safe(
        self,
        config: LocatorConfig,
        echoes: list[ElementHandle],
        *,
        customer: bool,
        controls: list[ElementHandle],
    ) -> tuple[bool, int]:
        handles = await locate(self.owned.locator, config, include_hidden=True).element_handles()
        try:
            if not handles or len(handles) > 120:
                return False, 0
            outcome: dict[str, Any] = await self.owned.root.evaluate(
                "(root, args) => {"
                + _HELPERS
                + """
                return {valid: args.nodes.every(node => safeLeaf(root, node) &&
                  readerActor(root, node) !== (args.customer ? 'assistant' : 'customer')) &&
                  !args.nodes.some((node, i) => args.nodes.some((other, j) =>
                    i !== j && overlap(node, other))) &&
                  (args.customer ? args.echoes.every(echo => args.nodes.includes(echo)) &&
                    args.nodes.filter(node => !args.before.includes(node))
                      .every(node => args.echoes.includes(node)) :
                    !args.nodes.some(node => args.echoes.some(echo =>
                      overlap(node, customerMessage(root, echo, args.before))))),
                owned: args.controls.filter(control => args.nodes.some(node =>
                  composedContains(node, control))).length};
                }""",
                {
                    "nodes": handles,
                    "echoes": echoes,
                    "customer": customer,
                    "before": self._before,
                    "controls": controls,
                },
            )
            return bool(outcome["valid"]), int(outcome["owned"])
        finally:
            for handle in handles:
                await handle.dispose()

    async def validate(self, assistant: LocatorConfig, customer: LocatorConfig) -> None:
        if self._text is None:
            raise ChannelError("channel.reader", "exchange baseline is missing")
        await self._assert_partition(assistant, customer, self._text, fresh=True)

    async def assert_partition(
        self, assistant: LocatorConfig, customer: LocatorConfig, text: str
    ) -> None:
        await self._assert_partition(assistant, customer, text, fresh=False)

    async def _assert_partition(
        self, assistant: LocatorConfig, customer: LocatorConfig, text: str, *, fresh: bool
    ) -> None:
        await self.owned.check()
        if customer is None or not normalize_echo(text):
            raise ChannelError("channel.reader", "separated customer reader and text are required")
        assistants = await locate(
            self.owned.locator, assistant, include_hidden=True
        ).element_handles()
        customers = await locate(
            self.owned.locator, customer, include_hidden=True
        ).element_handles()
        handles = assistants + customers
        try:
            if not assistants or not customers or len(assistants) > 120 or len(customers) > 120:
                raise ChannelError("channel.reader", "exchange readers are absent or oversized")
            data = await self.owned.root.evaluate(
                "(root, args) => {"
                + _HELPERS
                + """
                const all = [...args.assistants, ...args.customers];
                return {valid: all.every(node => safeLeaf(root, node)) &&
                  args.assistants.every(node => readerActor(root, node) !== 'customer') &&
                  args.customers.every(node => readerActor(root, node) !== 'assistant') &&
                  !all.some((node, i) => all.some((other, j) => i !== j && overlap(node, other))),
                  assistants: args.assistants.map(node => ({text: proseText(node),
                    fresh: !args.before.includes(node)})),
                  customers: args.customers.map(node => ({text: proseText(node),
                    fresh: !args.before.includes(node)}))};
                }""",
                {"assistants": assistants, "customers": customers, "before": self._before},
            )
            expected = normalize_echo(text)
            echoes = [
                item
                for item in data["customers"]
                if normalize_echo(item["text"]) == expected and (not fresh or item["fresh"])
            ]
            if (
                not data["valid"]
                or not echoes
                or (fresh and len(echoes) != 1)
                or any(normalize_echo(item["text"]) == expected for item in data["assistants"])
                or (
                    fresh
                    and any(
                        item["fresh"] and normalize_echo(item["text"]) != expected
                        for item in data["customers"]
                    )
                )
                or (fresh and not any(item["fresh"] for item in data["assistants"]))
            ):
                raise ChannelError("channel.reader", "exchange actor partition is ambiguous")
            if fresh:
                # A selected customer locator cannot hide a competing fresh echo.
                nodes, metadata = await self._inventory()
                actual_echoes = await self._fresh_echoes(nodes, metadata, text)
                if len(actual_echoes) != 1 or not await customers[0].evaluate(
                    "(node, args) => args.customers.includes(args.echo)",
                    {"customers": customers, "echo": actual_echoes[0]},
                ):
                    raise ChannelError("channel.reader", "fresh submitted-text echo is ambiguous")
                isolated = await self.owned.root.evaluate(
                    "(root, args) => {" + _HELPERS + "return !args.assistants.some(node => "
                    "overlap(node, customerMessage(root, args.echo, args.before)));}",
                    {"assistants": assistants, "echo": actual_echoes[0], "before": self._before},
                )
                if not isolated:
                    raise ChannelError(
                        "channel.reader", "response belongs to customer message container"
                    )
        finally:
            for handle in handles:
                await handle.dispose()
