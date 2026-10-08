"""Fresh native calibration browser: fixed DOM inspection and exactly one submission."""

from __future__ import annotations

import asyncio
from typing import Any, cast
from uuid import uuid4

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import JSHandle

from syntheticcx.adapters.channel.cursor import Actor, Bubble
from syntheticcx.adapters.channel.playwright import PlaywrightChannel, locate, normalize_echo
from syntheticcx.adapters.discovery.dom import DOM_HELPERS, TEXT_HELPER
from syntheticcx.adapters.discovery.playwright_mcp import _origin
from syntheticcx.config.models import ChannelConfig, ExecutionLimits, JsonObject, LocatorConfig
from syntheticcx.errors import ChannelError, ConfigurationError

# No page/model supplied executable code, input values, links, or raw HTML.
# Keep ancestor metadata so nested text spans can be mapped to message bubbles.
_INVENTORY = """(anchors, options) => {
  __DOM_HELPERS__
  const anchor = anchors[0];
  if (!anchor) return {status: 'missing_scope'};
  const doc = anchor.ownerDocument;
  const registry = options.registry;
  if (!registry || registry.document !== doc) return {status: 'registry_document_changed'};
  const nodeId = node => {
    if (!registry.nodes.has(node)) {
      registry.nodes.set(node, registry.prefix + '-' + ++registry.next);
    }
    return registry.nodes.get(node);
  };
  const origin = doc.location.origin;
  if (origin !== new URL(options.url).origin) return {origin, entries: []};
  let hint = options.hint;
  let root;
  if (!hint) {
    if (anchors.length !== 1) return {origin, status: 'missing_scope'};
    root = composedClosest(anchor, '[role=dialog], dialog') ||
      (anchor.getRootNode().host ? anchor.getRootNode() : doc.body);
    const address = scopeAddress(root, doc);
    const scan = walkDOM(root);
    if (!address || !scan.complete) return {origin, status: 'missing_scope'};
    const position = scan.nodes.indexOf(anchor);
    if (position < 0) return {origin, status: 'missing_scope'};
    hint = {address, position};
  } else {
    // A route enters only remembered open hosts; no cross-widget/body fallback.
    root = resolveScope(doc, hint.address, options.scope_nodes);
    if (!root) return {origin, status: 'missing_scope'};
  }
  const scan = walkDOM(root);
  if (!scan.complete) return {origin, status: 'missing_scope'};
  const all = scan.nodes;
  const position = hint.position;
  // Widgets are often appended after a large page. Inspect near the known input,
  // rather than truncating the page before ever reaching the chat.
  const nodes = all.slice(Math.max(0, position - 500), position + 500);
  const ranks = new Map(all.map((node, index) => [node, Math.abs(index - position)]));
  const visible = node => !!node.getClientRects().length;
  const attributes = node => Object.fromEntries(Array.from(node.attributes)
    .filter(a => ['id', 'class', 'role', 'aria-label'].includes(a.name) ||
      a.name.startsWith('data-'))
    .slice(0, 16).map(a => [a.name, a.value.slice(0, 160)]));
  const scope_candidates = hint.scopes || (() => {
    const result = []; let node = composedParent(anchor);
    while (node && node !== doc.body && result.length < 4) {
      result.push({tag: node.tagName.toLowerCase(), attributes: attributes(node)});
      node = composedParent(node);
    } return result;
  })();
  hint.scopes = scope_candidates;
  const entries = nodes.filter(node => visible(node) &&
    !composedClosest(node, 'input, textarea, select, script, style, [role=textbox], ' +
      '[contenteditable]:not([contenteditable=false])') &&
    !composedClosest(node, 'button, a') &&
    Array.from(node.childNodes).some(n => n.nodeType === 3 && n.textContent.trim()))
    .sort((left, right) => ranks.get(left) - ranks.get(right))
    .slice(0, 120).map(node => ({
      node_id: nodeId(node), tag: node.tagName.toLowerCase(), attributes: attributes(node),
      text: (node.innerText || '').slice(0, 1000),
      ancestors: (() => { const result = []; let parent = composedParent(node);
        while (parent && parent !== root && parent !== root.host && result.length < 4) {
          result.push({node_id: nodeId(parent), tag: parent.tagName.toLowerCase(),
            attributes: attributes(parent)});
          parent = composedParent(parent);
        } return result; })()
    }));
   const option_entries = nodes.filter(node => visible(node) &&
     node.matches('button, [role=button]') && !composedClosest(node, 'form'))
     .slice(0, 48).map(node => ({node_id: nodeId(node), tag: node.tagName.toLowerCase(),
       attributes: attributes(node),
       text: (node.innerText || node.getAttribute('aria-label') || '').slice(0, 160),
       ancestors: (() => { const result = []; let parent = composedParent(node);
         while (parent && parent !== root && parent !== root.host && result.length < 4) {
           result.push({node_id: nodeId(parent), tag: parent.tagName.toLowerCase(),
             attributes: attributes(parent)});
           parent = composedParent(parent);
         } return result; })()}));
   const scope_node = root.host || root;
   const scope_metadata = scope_node.tagName && scope_node !== doc.body ?
     {tag: scope_node.tagName.toLowerCase(), attributes: attributes(scope_node)} : null;
   const scope_nodes = [];
  if (resolveScope(doc, hint.address, options.scope_nodes, scope_nodes) !== root)
    return {origin, status: 'missing_scope'};
   return {origin, entries, option_entries, scope_metadata, scope_candidates,
     node_ids: all.map(nodeId),
     hint, scope_nodes};
}""".replace("__DOM_HELPERS__", DOM_HELPERS)

_INVENTORY_HANDLE = """(body, options) => {
  const result = (__INVENTORY__)(options.anchors, options);
  const scopes = result.scope_nodes;
  delete result.scope_nodes;
  return {result, scopes};
}""".replace("__INVENTORY__", _INVENTORY)

_ATTEST_COMPOSER_SCOPES = """(body, anchors) => {
  __DOM_HELPERS__
  if (anchors.length !== 1 || !anchors[0].isConnected || !renderedElement(anchors[0])) return [];
  const result = []; let node = composedParent(anchors[0]);
  while (node && !node.matches('body, html') && result.length < 4) {
    if (!node.matches('form') && renderedElement(node)) result.push(node);
    node = composedParent(node);
  }
  return result;
}""".replace("__DOM_HELPERS__", DOM_HELPERS)

_SELECTION_OWNERSHIP = """(nodes, options) => {
  __DOM_HELPERS__
  const origin = location.origin;
  if (origin !== new URL(options.url).origin) return {origin, status: 'wrong_origin'};
  const root = options.hint ? resolveScope(document, options.hint.address, options.scope_nodes) :
    document.body;
  if (!root) return {origin, status: 'missing_scope'};
  const owned = node => {
    if (!root.host) return composedContains(root, node);
    for (let current = node, count = 0; current && count < 20000;
      current = composedParent(current), count++) {
      if (current.getRootNode() === root) return true;
    }
    return false;
  };
  if (nodes.some(node => !owned(node))) return {origin, status: 'outside_scope'};
  if (options.registry && options.registry.document !== document)
    return {origin, status: 'registry_document_changed'};
  return {origin, status: 'owned'};
}""".replace("__DOM_HELPERS__", DOM_HELPERS)

_MESSAGE_EVIDENCE = """(nodes, options) => {
  __DOM_HELPERS__
  const origin = location.origin;
  if (origin !== new URL(options.url).origin) return {origin, entries: []};
  const ownership = (__SELECTION_OWNERSHIP__)(nodes, options);
  if (ownership.status !== 'owned') return {...ownership, entries: []};
  const evidenceRoot = options.hint ? resolveScope(document, options.hint.address,
    options.scope_nodes) : document.body;
  const attributes = node => Object.fromEntries(Array.from(node.attributes)
    .filter(a => ['id', 'class', 'role', 'aria-label'].includes(a.name) ||
      a.name.startsWith('data-'))
    .sort((a, b) => Number(b.name === 'id') - Number(a.name === 'id'))
    .slice(0, 16).map(a => [a.name, a.value.slice(0, 160)]));
  const metadata = node => ({tag: node.tagName.toLowerCase(), attributes: attributes(node)});
  const registry = options.registry;
  const nodeId = node => {
    if (!registry) return null;
    if (!registry.nodes.has(node)) {
      registry.nodes.set(node, registry.prefix + '-' + ++registry.next);
    }
    return registry.nodes.get(node);
  };
  return {origin, entries: nodes.slice(0, 120).map(node => {
    const scan = walkDOM(node);
    const contains_input =
      node.matches('input, textarea, [role=textbox], [contenteditable=true]') ||
      !scan.complete || scan.nodes.some(el =>
        el.matches('input, textarea, [role=textbox], [contenteditable=true]'));
    const unsafe = contains_input ||
      !!composedClosest(node, 'input, textarea, select, button, a, script, style, ' +
        '[role=textbox], ' +
        '[contenteditable]:not([contenteditable=false])');
    const ancestors = []; let parent = composedParent(node);
    while (!unsafe && parent && parent !== evidenceRoot && parent !== evidenceRoot.host &&
      !['BODY', 'HTML'].includes(parent.tagName) &&
      ancestors.length < 4) {
      ancestors.push(metadata(parent)); parent = composedParent(parent);
    }
    const text = unsafe ? '' : composedText(node, 4001, !!options.options_enabled);
    const buttons = options.options_enabled && !unsafe ? scan.nodes.filter(el =>
      el.matches('button, [role=button]') && renderedElement(el)) : [];
    return {
      text: (text || '').slice(0, 4000),
      truncated: text === null || text.length > 4000,
      identity: options.attribute ? node.getAttribute(options.attribute) :
        (options.identities.find(entry => entry.node === node)?.identity ?? null),
      visible: renderedElement(node), contains_input,
      node_id: nodeId(node),
      options: buttons.map(el => ({label:
        (el.innerText || el.getAttribute('aria-label') || '').slice(0, 160)})),
      tag: node.tagName.toLowerCase(), role: node.getAttribute('role'),
      attributes: unsafe ? {} : attributes(node), ancestors
    };
  })};
}""".replace("__DOM_HELPERS__", DOM_HELPERS + TEXT_HELPER).replace(
    "__SELECTION_OWNERSHIP__", _SELECTION_OWNERSHIP
)


class CalibrationObservationUnavailable(ChannelError):
    """Only recognized observation churn may be re-observed, never resubmitted."""

    def __init__(self, message: str) -> None:
        super().__init__("discover calibration", message, retryable=True)


class NativeCalibrationBrowser(PlaywrightChannel):
    """Reuse native control startup/send/cleanup, not guessed message tracking."""

    def __init__(
        self, config: ChannelConfig, limits: ExecutionLimits, *, headed: bool = False
    ) -> None:
        super().__init__(config, limits, headed=headed)
        self._observation_hint: JsonObject | None = None
        self._scope_nodes: JSHandle | None = None
        self._attested_composer_scopes: JSHandle | None = None
        self._node_registry: JSHandle | None = None
        self._anonymous_scope_lost = False
        self.option_validation_deferred = False

    def _check_origin(self, url: str | None) -> None:
        try:
            valid = url is not None and _origin(url) == _origin(str(self._config.url))
        except ConfigurationError:
            valid = False
        if not valid:
            raise ChannelError("discover navigation", "browser left the selected target origin")

    async def _snapshot(self, actor: Actor) -> list[Bubble]:
        # The provisional message locators/identity must not block control startup
        # or submission. Calibration observes the DOM independently instead.
        return []

    async def _prepare_completion(self) -> None:
        pass

    async def _ensure_node_registry(self) -> None:
        assert self._scope is not None
        if self._node_registry is None:
            self._node_registry = await self._scope.locator("body").evaluate_handle(
                "(body, prefix) => ({document: body.ownerDocument, "
                "nodes: new WeakMap(), next: 0, prefix})",
                uuid4().hex,
            )

    async def _drop_context_handles(self) -> None:
        """IDs cannot alias a new document; anonymous remembered scopes cannot be re-guessed."""
        if self._observation_hint is not None:
            address = self._observation_hint.get("address", [])
            self._anonymous_scope_lost |= any(
                isinstance(step, dict) and not step.get("shadow") and not step.get("id")
                for step in address
            )
        references = (self._node_registry, self._scope_nodes, self._attested_composer_scopes)
        self._node_registry = self._scope_nodes = self._attested_composer_scopes = None
        for handle in references:
            if handle is not None:
                try:
                    await handle.dispose()
                except PlaywrightError:
                    pass

    async def inventory(self) -> JsonObject:
        if self._scope is None:
            raise ChannelError("discover calibration", "chat is not started")
        if self._anonymous_scope_lost:
            raise CalibrationObservationUnavailable("anonymous chat observation scope was replaced")
        self._check_origin(self.current_url)
        try:
            if self._config.frame_locator is not None and self._page is not None:
                frames = await locate(self._page, self._config.frame_locator).count()
                if frames == 0:
                    raise CalibrationObservationUnavailable("chat observation frame is unavailable")
                if frames != 1:
                    raise ChannelError(
                        "discover calibration", "chat observation frame is ambiguous"
                    )
            locator = (
                locate(self._scope, self._config.input_locator)
                if self._observation_hint is None
                else self._scope.locator("body")
            )
            anchors = await locator.element_handles()
            handle = None
            try:
                await self._ensure_node_registry()
                if self._observation_hint is None and self._attested_composer_scopes is None:
                    self._attested_composer_scopes = await self._scope.locator(
                        "body"
                    ).evaluate_handle(_ATTEST_COMPOSER_SCOPES, anchors)
                handle = await self._scope.locator("body").evaluate_handle(
                    _INVENTORY_HANDLE,
                    {
                        "anchors": anchors,
                        "hint": self._observation_hint,
                        "url": str(self._config.url),
                        "scope_nodes": self._scope_nodes,
                        "registry": self._node_registry,
                    },
                )
                value = await handle.get_property("result")
                try:
                    result = cast(JsonObject, await value.json_value())
                finally:
                    await value.dispose()
                hint = result.get("hint")
                if self._scope_nodes is None and isinstance(hint, dict):
                    address = hint.get("address", [])
                    if not address or any(
                        isinstance(step, dict) and not step.get("shadow") and not step.get("id")
                        for step in address
                    ):
                        self._scope_nodes = await handle.get_property("scopes")
            finally:
                if handle is not None:
                    await handle.dispose()
                for anchor in anchors:
                    await anchor.dispose()
            self._check_origin(self.current_url)
            if "origin" in result:
                self._check_origin(str(result["origin"]))
            if result.get("status") == "missing_scope":
                raise CalibrationObservationUnavailable("chat observation scope is unavailable")
            if result.get("status") == "registry_document_changed":
                await self._drop_context_handles()
                raise CalibrationObservationUnavailable("chat observation document was replaced")
            hint = result.pop("hint", None)
            if isinstance(hint, dict):
                self._observation_hint = hint
            return result
        except PlaywrightError as error:
            self._check_origin(self.current_url)
            # Match only fixed Playwright classifications; never expose raw text.
            if any(
                reason in str(error)
                for reason in (
                    "Execution context was destroyed",
                    "JSHandles can be evaluated only in the context",
                    "JSHandles can be evaluated only in context",
                )
            ):
                await self._drop_context_handles()
                raise CalibrationObservationUnavailable(
                    "chat observation execution context was replaced"
                ) from None
            if any(
                reason in str(error)
                for reason in ("Frame was detached", "Failed to find frame for selector")
            ):
                await self._drop_context_handles()
                raise CalibrationObservationUnavailable(
                    "chat observation frame is unavailable"
                ) from None
            raise ChannelError(
                "discover calibration", "chat DOM inspection script failed"
            ) from None

    async def message_evidence(self, config: ChannelConfig) -> tuple[list[Any], list[Any]]:
        """Validate selectors without executing model-supplied inspection code."""
        if self._scope is None:
            raise ChannelError("discover calibration", "chat is not started")
        self._check_origin(self.current_url)
        stage = "message scope origin check"
        try:
            # Guard the frame before even count/overlap diagnostics can become corrections.
            self._check_origin(
                await self._scope.locator("body").evaluate_all("() => location.origin")
            )
            customer_locator = locate(self._scope, config.customer_message_locator)
            assistant_locator = locate(self._scope, config.assistant_message_locator)
            stage = "customer message locator"
            customer_count = await customer_locator.count()
            stage = "assistant message locator"
            assistant_count = await assistant_locator.count()
            if max(customer_count, assistant_count) > 120:
                raise ChannelError("discover calibration", "message inspection limit exceeded")
            if self._anonymous_scope_lost:
                raise CalibrationObservationUnavailable(
                    "anonymous chat observation scope was replaced"
                )
            await self._ensure_node_registry()
            ownership_options = {
                "url": str(config.url),
                "hint": self._observation_hint,
                "scope_nodes": self._scope_nodes,
                "registry": self._node_registry,
            }
            # Reject out-of-scope selections before inspecting overlap, attributes,
            # option prose, or text. Neither model corrections nor callers see it.
            for locator in (customer_locator, assistant_locator):
                owned = await locator.evaluate_all(_SELECTION_OWNERSHIP, ownership_options)
                self._check_origin(owned["origin"])
                if owned["status"] == "registry_document_changed":
                    await self._drop_context_handles()
                    raise CalibrationObservationUnavailable(
                        "chat observation document was replaced"
                    )
                if owned["status"] != "owned":
                    raise ChannelError(
                        "discover calibration",
                        "message selector leaves the owned observation scope",
                    )
            stage = "customer message node lookup"
            handles = await customer_locator.element_handles()
            try:
                stage = "message container overlap check"
                overlaps = await assistant_locator.evaluate_all(
                    "(nodes, customers) => {"
                    + DOM_HELPERS
                    + " return nodes.some(node => customers.some(customer => "
                    "composedContains(node, customer) || composedContains(customer, node))); }",
                    handles,
                )
            finally:
                for handle in handles:
                    await handle.dispose()
            if overlaps:
                raise ChannelError(
                    "discover calibration",
                    "assistant selector overlaps customer message containers",
                )
            option_evidence: list[dict[str, Any]] = []
            if config.reply_options is not None:
                await self._verify_option_scope(config)
                composer_count = await locate(self._scope, config.input_locator).count()
                self.option_validation_deferred = composer_count == 0
                if not self.option_validation_deferred:
                    option_evidence = await self.inspect_reply_options(config)
                if not self.option_validation_deferred and not any(
                    option.get("eligible")
                    for reply in option_evidence
                    if reply.get("latest")
                    for option in reply.get("options", [])
                ):
                    raise ChannelError(
                        "discover calibration",
                        "reply options lack guarded informational menu evidence",
                    )
            stage = "message node identity inspection"
            customer_ids, assistant_ids = (
                await asyncio.gather(
                    self._node_identities("customer", customer_locator),
                    self._node_identities("assistant", assistant_locator),
                )
                if config.message_identity.strategy == "dom_node"
                else ([], [])
            )
            stage = "message text inspection"
            customer, assistant = await asyncio.gather(
                customer_locator.evaluate_all(
                    _MESSAGE_EVIDENCE,
                    {
                        **ownership_options,
                        "attribute": config.message_identity.attribute_name,
                        "url": str(config.url),
                        "identities": customer_ids,
                        "options_enabled": False,
                    },
                ),
                assistant_locator.evaluate_all(
                    _MESSAGE_EVIDENCE,
                    {
                        **ownership_options,
                        "attribute": config.message_identity.attribute_name,
                        "url": str(config.url),
                        "identities": assistant_ids,
                        "options_enabled": config.reply_options is not None,
                    },
                ),
            )
            self._check_origin(self.current_url)
            self._check_origin(customer["origin"])
            self._check_origin(assistant["origin"])
            if any(value.get("status") not in {None, "owned"} for value in (customer, assistant)):
                raise ChannelError(
                    "discover calibration", "message selector leaves the owned observation scope"
                )
            if config.reply_options is not None and not self.option_validation_deferred:
                if len(option_evidence) != len(assistant["entries"]):
                    raise ChannelError(
                        "discover calibration", "reply option wrapper evidence changed"
                    )
                for entry, reply in zip(assistant["entries"], option_evidence, strict=True):
                    entry["options"] = [
                        option for option in reply["options"] if option.get("eligible")
                    ]
            elif config.reply_options is not None:
                # Only an unverified candidate is returned. Fresh verification binds the
                # channel helper before sending, and must prove eligibility on both replies.
                for index, entry in enumerate(assistant["entries"]):
                    entry["options"] = await locate(
                        assistant_locator.nth(index), config.reply_options.option_locator
                    ).evaluate_all(
                        "nodes => nodes.map(node => ({label: "
                        "(node.innerText || node.getAttribute('aria-label') || '').slice(0, 160), "
                        "ownership_validation: 'deferred_to_fresh_verification'}))"
                    )
            return customer["entries"], assistant["entries"]
        except PlaywrightError as error:
            self._check_origin(self.current_url)
            if any(
                marker in str(error)
                for marker in (
                    "Execution context was destroyed",
                    "JSHandles can be evaluated only in the context",
                    "JSHandles can be evaluated only in context",
                    "Frame was detached",
                    "Failed to find frame for selector",
                )
            ):
                await self._drop_context_handles()
                raise CalibrationObservationUnavailable(
                    "chat observation execution context was replaced"
                ) from None
            # Classify only known parser errors; never include raw SDK text or selectors.
            parser_error = any(
                marker in str(error)
                for marker in (
                    "Unexpected token",
                    "Unknown engine",
                    "Unsupported token",
                    "while parsing",
                    "is not a valid selector",
                    "InvalidSelectorError",
                )
            )
            raise ChannelError(
                "discover calibration",
                f"{stage} has invalid or unsupported selector syntax"
                if parser_error
                else f"message selector inspection failed during {stage}",
            ) from None

    async def locator_matches(self, locator: LocatorConfig) -> int:
        """Read-only syntax/count validation before exposing an owned catalog candidate."""
        if self._scope is None:
            return 0
        self._check_origin(self.current_url)
        try:
            self._check_origin(
                await self._scope.locator("body").evaluate_all("() => location.origin")
            )
            return int(await locate(self._scope, locator).count())
        except PlaywrightError:
            return 0

    async def reading_candidate_matches(
        self,
        locator: LocatorConfig,
        kind: str,
        baseline: JsonObject,
        after: JsonObject,
        probe: str,
    ) -> int:
        """Offer only scoped, actor-separated reading candidates, without model guesses."""
        if self._scope is None or self._observation_hint is None:
            return 0
        previous = set(baseline.get("node_ids", []))
        echo_ids = [
            entry["node_id"]
            for entry in after.get("entries", [])
            if isinstance(entry, dict)
            and entry.get("node_id") not in previous
            and normalize_echo(str(entry.get("text", ""))) == normalize_echo(probe)
        ]
        await self._ensure_node_registry()
        options: dict[str, Any] = {
            "url": str(self._config.url),
            "hint": self._observation_hint,
            "scope_nodes": self._scope_nodes,
            "registry": self._node_registry,
            "attribute": None,
            "identities": [],
            "options_enabled": True,
        }
        self._check_origin(self.current_url)
        try:
            selected = locate(self._scope, locator, include_hidden=True)
            count = await selected.count()
            if not 0 < count <= 120:
                return 0
            evidence = await selected.evaluate_all(_MESSAGE_EVIDENCE, options)
            self._check_origin(evidence["origin"])
            if evidence.get("status") not in {None, "owned"}:
                return 0
            entries = evidence["entries"]
            if any(
                entry["contains_input"]
                or entry["truncated"]
                or entry["tag"] in {"body", "html", "button", "a", "form"}
                or entry.get("role") in {"dialog", "log", "status", "progressbar"}
                for entry in entries
            ):
                return 0
            if kind == "customer":
                fresh = [entry for entry in entries if entry["node_id"] not in previous]
                return (
                    count
                    if fresh
                    and all(
                        entry["visible"] and normalize_echo(entry["text"]) == normalize_echo(probe)
                        for entry in fresh
                    )
                    else 0
                )
            # Compare actual selected nodes with fresh echo leaves, including containment.
            # Shared wrapper classes and transcript ancestors never enter the assistant catalog.
            separated = await selected.evaluate_all(
                "(nodes, options) => {"
                + DOM_HELPERS
                + """
                  const root = resolveScope(document, options.hint.address, options.scope_nodes);
                  if (!root) return false;
                  const scan = walkDOM(root);
                  if (!scan.complete) return false;
                  const echoes = scan.nodes.filter(node => options.echo_ids.includes(
                    options.registry.nodes.get(node)));
                  return echoes.length === options.echo_ids.length &&
                    nodes.every(node => node !== root && !echoes.some(echo =>
                      composedContains(node, echo) || composedContains(echo, node)));
                }""",
                {**options, "echo_ids": echo_ids},
            )
            self._check_origin(self.current_url)
            return count if separated else 0
        except PlaywrightError:
            self._check_origin(self.current_url)
            return 0

    async def _verify_option_scope(self, config: ChannelConfig) -> None:
        """Pin proof to actual pre-send composer ancestors; never use body or replacement roots."""
        assert self._scope is not None and config.reply_options is not None
        scope = locate(self._scope, config.reply_options.scope_locator)
        if await scope.count() != 1:
            raise ChannelError("discover calibration", "reply option scope is ambiguous or missing")
        handles = await locate(self._scope, config.input_locator).element_handles()
        wrappers = await locate(self._scope, config.assistant_message_locator).element_handles()
        try:
            valid = await scope.evaluate_all(
                "(nodes, owned) => {"
                + DOM_HELPERS
                + 'const scope = nodes[0]; return scope && !scope.matches("body, html") && '
                "scope.isConnected && renderedElement(scope) && "
                "owned.attested && owned.attested.includes(scope) && "
                "owned.composers.length <= 1 && owned.wrappers.length > 0 && "
                "[...owned.composers, ...owned.wrappers].every(node => "
                "composedContains(scope, node));}",
                {
                    "composers": handles,
                    "wrappers": wrappers,
                    "attested": self._attested_composer_scopes,
                },
            )
        finally:
            for handle in [*handles, *wrappers]:
                await handle.dispose()
        if not valid:
            raise ChannelError(
                "discover calibration", "reply options lack verified conversation ownership"
            )
        # Options are resolved relative to each assistant wrapper, never to the page.
        assistants = locate(self._scope, config.assistant_message_locator)
        found = 0
        count = await assistants.count()
        for index in range(count):
            options = locate(assistants.nth(index), config.reply_options.option_locator)
            found += await options.count()
            if found > 48:
                raise ChannelError("discover calibration", "reply option inspection limit exceeded")
            if index != count - 1:
                continue  # Historical menus may legitimately be disabled; never enable/select them.
            if not await options.evaluate_all(
                "(nodes) => {" + DOM_HELPERS + "return nodes.every(node => "
                'node.matches("button, [role=button]") && !composedClosest(node, "form, a") && '
                '!node.matches(":disabled") && node.getAttribute("aria-disabled") !== "true" && '
                "renderedElement(node));}",
            ):
                raise ChannelError(
                    "discover calibration", "reply options include unsafe or unavailable controls"
                )
        if not found:
            raise ChannelError(
                "discover calibration", "reply option selector matched no assistant-owned controls"
            )

    async def close(self) -> None:
        handle, self._scope_nodes = self._scope_nodes, None
        attested, self._attested_composer_scopes = self._attested_composer_scopes, None
        registry, self._node_registry = self._node_registry, None
        self._observation_hint = None
        self._anonymous_scope_lost = False
        self.option_validation_deferred = False
        try:
            for reference in (handle, attested, registry):
                if reference is None:
                    continue
                try:
                    await reference.dispose()
                except PlaywrightError:
                    pass  # A detached/destroyed document already discarded its references.
        finally:
            await super().close()
