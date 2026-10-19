"""Bounded native reply-option inspection and one-use, node-owned capabilities."""

from __future__ import annotations

import re
import secrets
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from playwright.async_api import ElementHandle, FrameLocator, Locator, Page
from playwright.async_api import Error as PlaywrightError

from syntheticcx.adapters.channel.reading import READING_HELPERS
from syntheticcx.config.models import ChannelConfig, LocatorConfig
from syntheticcx.discovery.policy import portability_flow
from syntheticcx.domain.models import ReplyOption
from syntheticcx.errors import ChannelError


def normalized_label(value: str) -> str:
    folded = unicodedata.normalize("NFKC", value).casefold()
    clean = "".join(
        " " if char.isspace() else char
        for char in folded
        if char.isspace() or unicodedata.category(char) not in {"Cf", "Cc", "Cs"}
    )
    return " ".join(clean.split())


def _words(value: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", normalized_label(value))
        if not unicodedata.combining(char)
    )


_UNSAFE = re.compile(
    r"\b(pay(?:ment)?|purchase|buy|checkout|order|subscribe|consent|agree|accept|"
    r"authorize|password|login|log in|sign in|sign up|register|authenticate|upload|download|delete|"
    r"cancel|transfer|confirm|pagar|pagamento|comprar|compra|contratar|assinar|"
    r"consentimento|aceitar|concordar|autorizar|senha|entrar|autenticar|enviar arquivo|"
    r"excluir|cancelar|transferir|confirmar|cartao|credit card|bank account|cpf|"
    r"cadastro|cadastre)\b"
)
_ACTION = re.compile(
    r"\b(activate|activation|deactivate|change|modify|upgrade|downgrade|enable|disable|"
    r"withdraw|withdrawal|send|deposit|redeem|renew|terminate|book|reserve|install|"
    r"switch|update|apply|add|remove|manage|ativar|ativacao|desativar|alterar|alteracao|"
    r"mudar|modificar|habilitar|desabilitar|sacar|saque|enviar|depositar|resgatar|"
    r"renovar|encerrar|reservar|instalar|trocar|atualizar|adicionar|remover|gerenciar)\b"
)
_MENU = re.compile(
    r"\b(choose|select|pick|which|what|menu|topics?|information|learn|help|"
    r"escolha|escolher|selecione|selecionar|qual|quais|menu|assuntos?|temas?|"
    r"informacoes|informacao|saber|ajuda|conhecer)\b"
)
_INFO = re.compile(
    r"\b(plans?|pricing|prices?|features?|coverage|services?|support|billing|"
    r"invoices?|hours|locations?|faq|help|details?|benefits?|internet|mobile|"
    r"planos?|precos?|valores?|recursos?|cobertura|servicos?|suporte|fatura|"
    r"horarios?|enderecos?|ajuda|detalhes?|beneficios?|telefonia|contas?|faturas?)\b"
)
# Entire label vocabulary, not a substring match: an informational surrounding
# prompt must never authorize an arbitrary command such as "Activate plan".
_LABEL_WORDS = set(
    "plan plans pricing price prices feature features coverage service services support "
    "billing invoice invoices hours location locations faq help detail details benefit benefits "
    "internet mobile planos plano preco precos valor valores recurso recursos cobertura "
    "servico servicos suporte fatura faturas horario horarios endereco enderecos ajuda detalhe "
    "detalhes beneficio beneficios telefonia conta contas information info topic topics "
    "view read show see learn about more with for of and the my your customer "
    "compare check information details informacao informacoes tema temas assunto assuntos "
    "ver ler mostrar saber saiba mais sobre conhecer consultar visualizar comparar verificar "
    "com para de do da dos das e o a os as meu minha meus minhas duvida duvidas".split()
)


def informative_label(label: str) -> bool:
    """Require meaningful, positively recognized read-only EN/PT topic phrasing."""
    if any(
        unicodedata.category(char) in {"Cf", "Cc", "Cs"} and not char.isspace() for char in label
    ):
        return False
    choice = _words(label)
    words = re.findall(r"\w+", choice)
    return bool(
        words
        and _INFO.search(choice)
        and not _UNSAFE.search(choice)
        and not _ACTION.search(choice)
        and not _AMBIGUOUS.search(choice)
        and all(word in _LABEL_WORDS for word in words)
    )


_AMBIGUOUS = re.compile(
    r"^(yes|no|ok|okay|continue|next|confirm|submit|send|sim|nao|continuar|"
    r"proximo|avancar|confirmar|enviar|aceito)\b"
)

# Explicit native choice prompt in the owning reply prose (accent-folded by _words).
_CHOICE_PROMPT = re.compile(
    r"\b(?:escolha|escolhe|selecione|seleciona|selecionar|escolher)\s+"
    r"(?:\w+\s+){0,3}(?:opcao|opcoes)\b"
    r"|\b(?:choose|select|pick)\s+(?:\w+\s+){0,3}options?\b"
)
# Bubble utility controls are never conversational routing choices.
_UTILITY = re.compile(
    r"\b(copy|copiar|share|compartilhar|rate|avaliar|avaliacao|like|dislike|"
    r"thumbs|feedback|listen|ouvir|regenerate|regenerar|retry|tente novamente|"
    r"tentar novamente|read aloud|ler em voz alta|print|imprimir)\b"
)


_CONTACT_FIELD = r"(?:e-?mail|email address|telefone|phone(?: number)?)"
_OPTIONAL_CONTACT = re.compile(
    rf"\b{_CONTACT_FIELD}\b.*\b(?:opcional|optional|nao (?:e )?obrigatorio|not required)\b"
    rf"|\b(?:opcional|optional)\b.*\b{_CONTACT_FIELD}\b"
    rf"|\b(?:sem|without)\s+(?:(?:informar|fornecer|preencher|provide|providing|entering)\s+)?"
    rf"(?:(?:o|seu|um|your|an?)\s+)?{_CONTACT_FIELD}\b"
    rf"|\b(?:se nao quiser|if you (?:do not|don't) want to)\s+"
    rf"(?:informar|fornecer|preencher|provide|enter)\s+"
    rf"(?:(?:o|seu|um|your|an?)\s+)?{_CONTACT_FIELD}\b"
)
_NAVIGATION = re.compile(
    r"\b(?:navigate|redirect|external|website|site|link|url|redirecionar|navegar)\b"
)


def eligibility(
    prose: str,
    label: str,
    metadata: dict[str, Any],
    menu_labels: list[str] | None = None,
    *,
    dynamic_runtime: bool = False,
    authorizations: list[dict[str, Any]] | None = None,
) -> str | None:
    """Fail closed: positive local informational/menu evidence in EN or PT."""
    if not metadata.get("safe"):
        return "unsafe_control"
    if not metadata.get("visible") or not metadata.get("enabled"):
        return "unavailable_control"
    context, choice = _words(prose), _words(label)
    if not dynamic_runtime and portability_flow(prose, label):
        return "portability_forbidden_in_discovery"
    if dynamic_runtime:
        if _UTILITY.search(choice):
            return "utility_control"
        combined = f"{context} {choice}"
        categories = (
            (
                "payment",
                r"\b(pay|payment|charge|transfer|pagar|pagamento|cobrar|cobranca|transferir)\b",
            ),
            (
                "purchase",
                r"\b(purchase|buy|checkout|order|subscribe|comprar|compra|contratar|assinar)\b",
            ),
            (
                "account_change",
                r"\b(change|modify|upgrade|downgrade|activate|activation|deactivate|enable|"
                r"disable|switch|update|add|manage|renew|ativar|ativacao|desativar|habilitar|"
                r"desabilitar|alterar|alteracao|modificar|atualizar|mudar|trocar|adicionar|"
                r"gerenciar|renovar)\b",
            ),
            (
                "consent",
                r"\b(consent|agree|accept|authorize|consentimento|aceitar|concordar|autorizar|aceito)\b",
            ),
            (
                "authentication",
                r"\b(password|login|sign in|sign up|register|authenticate|senha|"
                r"entrar|autenticar|cadastro)\b",
            ),
            (
                "destructive",
                r"\b(delete|cancel|remove|terminate|excluir|cancelar|remover|encerrar)\b",
            ),
        )
        actions = {name for name, expression in categories if re.search(expression, combined)}
        permissions = {item.get("action"): item for item in authorizations or []}
        if any(
            not any(re.fullmatch(expression, match.group()) for _, expression in categories)
            for match in _ACTION.finditer(combined)
        ):
            return "scenario_action_not_authorized"
        if not actions:
            return (
                "scenario_action_not_authorized"
                if (_ACTION.search(combined) or _UNSAFE.search(combined))
                else None
            )
        if not actions.issubset(permissions):
            return "scenario_action_not_authorized"
        if "payment" in actions:
            authorization = permissions["payment"]
            if authorization.get("environment") != "sandbox":
                return "payment_requires_sandbox"
            # Prefer the selected control's amount, never the first price in a reply.
            pattern = (
                r"(?:\$|\b(?:[A-Z]{3}|(?i:usd|brl|eur|gbp|cad|aud|jpy|cny|chf|inr))\s*)"
                r"(-?\d[\d.,]*)"
            )
            if re.search(r"\d", label) and not re.search(pattern, label):
                # A bare selected amount must not borrow a smaller contextual price.
                return "payment_amount_unbounded"
            amount_source = label if re.search(pattern, label) else prose
            amounts = list(re.finditer(pattern, amount_source))
            if (
                len(amounts) != 1
                or len(re.findall(r"-?\d[\d.,]*", amount_source)) != 1
                or not re.fullmatch(r"\d+(?:[.,]\d{1,2})?", amounts[0].group(1))
                or re.match(r"\s+\d", amount_source[amounts[0].end() :])
            ):
                return "payment_amount_unbounded"
            value = float(amounts[0].group(1).replace(",", "."))
            if value > float(authorization["max_amount"]):
                return "payment_limit_exceeded"
            mentioned = {
                code.casefold()
                for pair in re.findall(r"\b([A-Z]{3})\s*-?\d|\d\s*([A-Z]{3})\b", f"{prose} {label}")
                for code in pair
                if code
            }
            mentioned.update(
                code.casefold() for code in re.findall(r"\b[A-Z]{3}\b", f"{prose} {label}")
            )
            mentioned.update(
                re.findall(r"\b(?:usd|brl|eur|gbp|cad|aud|jpy|cny|chf|inr)\b", combined)
            )
            if re.search(r"\b(?:reais|real)\b|r\$", combined):
                mentioned.add("brl")
            if re.search(r"\b(?:dolar|dollars?)\b|(?<!r)\$", combined):
                mentioned.add("usd")
            if mentioned != {str(authorization["currency"]).casefold()}:
                return "payment_currency_mismatch"
        # Keep ordinary conversational choices available; consequential choices
        # require explicit scenario authorization above.
        return None
    # Only exact continuation labels with explicit local optional-contact
    # evidence qualify. This exception never bypasses native or unsafe guards.
    optional_continue = (
        choice in {"continuar", "continue"}
        and not any(
            unicodedata.category(char) in {"Cf", "Cc", "Cs"} and not char.isspace()
            for char in label
        )
        and bool(_OPTIONAL_CONTACT.search(context))
    )
    if not choice or len(label) > 1000 or (_AMBIGUOUS.search(choice) and not optional_continue):
        return "ambiguous_label"
    if _UNSAFE.search(context + " " + choice) or _ACTION.search(context + " " + choice):
        return "unsafe_context"
    if _UTILITY.search(choice):
        return "utility_control"
    if optional_continue:
        return "unsafe_context" if _NAVIGATION.search(context) else None
    if _CHOICE_PROMPT.search(context):
        # An explicit native choice prompt makes reply-owned buttons conversational
        # routing options; the structural, transactional and utility guards above
        # still apply, and duplicate labels remain rejected by the caller.
        return None
    if not informative_label(label):
        return "no_informational_topic_label"
    # An options-only reply can itself be an informational menu, but only with
    # multiple distinct, positively recognized informational topic labels.
    labels = [_words(value) for value in menu_labels or []]
    label_menu = (
        not context.strip()
        and len(set(labels)) >= 2
        and all(informative_label(value) for value in menu_labels or [])
    )
    if not label_menu and (
        not _MENU.search(context) or not (_INFO.search(context) or _INFO.search(choice))
    ):
        return "no_informational_menu_evidence"
    return None


@dataclass
class OptionNode:
    node: ElementHandle
    label: str
    metadata: dict[str, Any]
    reason: str | None


@dataclass
class ReplyNodes:
    node: ElementHandle
    ordinal: int
    prose: str
    options: list[OptionNode]


@dataclass
class Capability:
    reply: ReplyNodes
    option: OptionNode
    message_id: str
    generation: int
    signature: tuple[Any, ...]


class ReplyOptions:
    """Pin scope/composer nodes, never fall back to a global or another frame."""

    def __init__(
        self,
        scope: Page | FrameLocator,
        config: ChannelConfig,
        locate: Callable[..., Locator],
        *,
        attested_scope: Any | None = None,
        dynamic_runtime: bool = False,
        authorizations: list[dict[str, Any]] | None = None,
    ) -> None:
        self.scope = scope
        self.config = config
        self.locate = locate
        self.root: ElementHandle | None = None
        self.composer: ElementHandle | None = None
        self.registry: dict[str, Capability] = {}
        self.generation = 0
        self._known: list[ElementHandle] = []
        self._attested_scope = attested_scope
        self._dynamic_runtime = dynamic_runtime
        self._authorizations = authorizations or []

    def invalidate(self) -> None:
        self.registry.clear()
        self.generation += 1

    async def check_origin(self) -> None:
        """Check parent URL and the selected document's security origin, value-free.

        Window.origin (rather than location.origin) also correctly checks inherited
        same-origin srcdoc documents, while rejecting opaque sandboxed documents.
        """
        if self._attested_scope is not None:
            await self._attested_scope.check()
            return
        page = self.scope.owner.page if isinstance(self.scope, FrameLocator) else self.scope

        def origin(url: str) -> tuple[str, str | None, int | None]:
            parsed = urlsplit(url)
            port = parsed.port
            return (
                parsed.scheme,
                parsed.hostname,
                (port if port is not None else (443 if parsed.scheme == "https" else 80)),
            )

        try:
            expected = origin(str(self.config.url))
            valid_parent = origin(page.url) == expected
            valid_document = valid_parent and await self.scope.locator("body").evaluate_all(
                "(_, expected) => globalThis.origin === new URL(expected).origin",
                str(self.config.url),
            )
            valid_parent = origin(page.url) == expected
        except (ValueError, PlaywrightError):
            self.invalidate()
            raise ChannelError("channel.options", "selected target origin is unavailable") from None
        if not valid_parent or not valid_document:
            self.invalidate()
            raise ChannelError("channel.options", "browser left the selected target origin")

    async def dispose(self) -> None:
        """Release inspection references without changing the DOM."""
        self.invalidate()
        for handle in [
            self.root if self._attested_scope is None else None,
            self.composer,
            *self._known,
        ]:
            if handle is not None:
                await handle.dispose()
        self._known.clear()

    async def _unique(self, config: LocatorConfig) -> ElementHandle:
        locator = self.locate(self.scope, config, include_hidden=True)
        if await locator.count() != 1:
            raise ChannelError("channel.options", "configured option scope/composer is ambiguous")
        handle = await locator.element_handle()
        if handle is None:
            raise ChannelError("channel.options", "configured option scope/composer is absent")
        return handle

    async def bind(self) -> None:
        config = self.config.reply_options
        assert config is not None
        await self.check_origin()
        if self._attested_scope is not None:
            self.root = self._attested_scope.root
            self.composer = (
                await self._unique(self.config.input_locator)
                if self.config.input_locator is not None
                else None
            )
            if self.composer is not None and not await self._attested_scope.contains(
                [self.composer]
            ):
                raise ChannelError("channel.options", "composer is outside attested root")
            await self.collect()
            return
        self.root = await self._unique(config.scope_locator)
        self.composer = await self._unique(self.config.input_locator)
        initial_valid = await self.root.evaluate(
            "(root, composer) => {"
            + READING_HELPERS
            + """
              return root.isConnected && composer.isConnected &&
                composedContains(root, composer);
            }""",
            self.composer,
        )
        if not initial_valid:
            self.invalidate()
            raise ChannelError("channel.options", "initial composer is outside the chat scope")
        await self.collect()

    async def _identity(self, node: ElementHandle) -> int:
        index: int = await node.evaluate("(node, known) => known.indexOf(node)", self._known)
        if index < 0:
            if len(self._known) >= 2000:
                raise ChannelError("channel.options", "option node history limit exceeded")
            index = len(self._known)
            self._known.append(node)
        return index

    async def collect(self) -> list[ReplyNodes]:
        config = self.config.reply_options
        assert config is not None and self.root is not None
        await self.check_origin()
        root = (
            self.root
            if self._attested_scope is not None
            else await self._unique(config.scope_locator)
        )
        composer_locator = (
            self.locate(self.scope, self.config.input_locator, include_hidden=True)
            if self.config.input_locator is not None
            else None
        )
        composer_count = await composer_locator.count() if composer_locator is not None else 0
        if composer_count > 1:
            self.invalidate()
            raise ChannelError("channel.options", "current composer is ambiguous")
        composer = (
            await composer_locator.element_handle()
            if composer_count and composer_locator is not None
            else None
        )
        wrappers = self.locate(
            self.scope, self.config.assistant_message_locator, include_hidden=True
        )
        handles = await wrappers.element_handles()
        customers = await self.locate(
            self.scope, self.config.customer_message_locator, include_hidden=True
        ).element_handles()
        send_controls = (
            await self.locate(
                self.scope, self.config.send_strategy.locator, include_hidden=True
            ).element_handles()
            if self.config.send_strategy.locator is not None
            else []
        )
        if len(handles) > 120 or len(customers) > 120 or len(send_controls) > 64:
            raise ChannelError("channel.options", "assistant inspection limit exceeded")
        valid = await root.evaluate(
            "(root, args) => {"
            + READING_HELPERS
            + """
              const invalidWrapper = 'html, body, button, a, form, input, textarea, select, ' +
                'dialog, ul, ol, nav, main, header, footer, output, progress, meter, ' +
                '[role=button], [role=link], [role=dialog], [role=alertdialog], ' +
                '[role=toolbar], [role=feed], [role=list], [role=listbox], [role=combobox], ' +
                '[role=menu], [role=menubar], ' +
                '[role=status], [role=log], [role=progressbar], [role=textbox], ' +
                '[contenteditable]:not([contenteditable=false])';
              const composerValid = !args.composer ||
                (args.composer === args.initialComposer && args.composer.isConnected &&
                  composedContains(root, args.composer));
               const forbidden = [...args.customers, ...args.sendControls,
                 ...(args.initialComposer ? [args.initialComposer] : []),
                 ...(args.composer ? [args.composer] : [])];
              return root === args.root && composerValid &&
                root !== root.ownerDocument.body && root.tagName !== 'HTML' &&
                root.isConnected && walkDOM(root).complete &&
                args.wrappers.every(node => node !== root && node.isConnected &&
                  composedContains(root, node) && !node.matches(invalidWrapper) &&
                  !forbidden.some(other => composedContains(node, other) ||
                    composedContains(other, node)) &&
                  !args.wrappers.some(other => other !== node &&
                    (composedContains(node, other) || composedContains(other, node))));
            }""",
            {
                "root": self.root,
                "composer": composer,
                "initialComposer": self.composer,
                "wrappers": handles,
                "customers": customers,
                "sendControls": send_controls,
            },
        )
        if not valid:
            self.invalidate()
            raise ChannelError("channel.options", "verified chat scope or ownership changed")
        if self._attested_scope is None:
            await root.dispose()
        if composer is not None:
            await composer.dispose()
        for handle in [*customers, *send_controls]:
            await handle.dispose()
        result: list[ReplyNodes] = []
        for ordinal, handle in enumerate(handles):
            owner_identity = await self._identity(handle)
            owner = self._known[owner_identity]
            prose: str = await handle.evaluate(
                "node => {" + READING_HELPERS + "return proseText(node); }"
            )
            options = await self.locate(
                wrappers.nth(ordinal), config.option_locator, include_hidden=True
            ).element_handles()
            if len(options) > 64:
                raise ChannelError("channel.options", "option inspection limit exceeded")
            nodes: list[OptionNode] = []
            for option in options:
                metadata: dict[str, Any] = await option.evaluate(
                    "(node, args) => {"
                    + READING_HELPERS
                    + """
                      const ancestry = [];
                      for (let p = node; p; p = p.assignedSlot || composedParent(p))
                        ancestry.push(p);
                      const navAttrs = new Set(['href', 'target', 'download', 'formaction',
                        'form', 'src', 'action', 'data-href', 'data-url', 'data-action']);
                      const unsafe = ancestry.some(p => p.matches('a, form, input, select, ' +
                        'textarea, [role=link], [contenteditable]:not([contenteditable=false])') ||
                        [...p.attributes].some(a => navAttrs.has(a.name.toLowerCase())));
                      const label = composedText(node, 1001);
                      const owners = args.wrappers.filter(p => composedContains(p, node));
                      return {label, connected: node.isConnected,
                        visible: renderedElement(node) && ancestry.every(p => {
                          const style = getComputedStyle(p);
                          return style.display !== 'none' && style.visibility === 'visible' &&
                            style.contentVisibility !== 'hidden' && style.opacity !== '0' &&
                            (p.getAttribute('aria-hidden') || '').toLowerCase() !== 'true';
                        }),
                        enabled: !node.matches(':disabled') && !ancestry.some(p =>
                          p.matches('fieldset[disabled], [disabled], [inert]') ||
                          (p.getAttribute('aria-disabled') || '').toLowerCase() === 'true'),
                        safe: !unsafe && (node.tagName === 'BUTTON' ?
                          !['submit', 'reset'].includes(
                            (node.getAttribute('type') || '').toLowerCase()) :
                          node.getAttribute('role') === 'button') &&
                          node.isConnected && composedContains(args.root, node) &&
                          owners.length === 1 && owners[0] === args.owner,
                        tag: node.tagName, role: node.getAttribute('role'),
                        type: node.getAttribute('type')};
                    }""",
                    {"root": self.root, "owner": handle, "wrappers": handles},
                )
                label = metadata.pop("label")
                if label is None:
                    raise ChannelError("channel.options", "incomplete option label scan")
                metadata["node_identity"] = await self._identity(option)
                retained = self._known[metadata["node_identity"]]
                if retained is not option:
                    await option.dispose()
                metadata["owner_identity"] = owner_identity
                nodes.append(
                    OptionNode(
                        retained,
                        label,
                        metadata,
                        eligibility(
                            prose,
                            label,
                            metadata,
                            dynamic_runtime=self._dynamic_runtime,
                            authorizations=self._authorizations,
                        ),
                    )
                )
            labels = [normalized_label(node.label) for node in nodes]
            for node in nodes:
                node.reason = eligibility(
                    prose,
                    node.label,
                    node.metadata,
                    [entry.label for entry in nodes],
                    dynamic_runtime=self._dynamic_runtime,
                    authorizations=self._authorizations,
                )
            if len(labels) != len(set(labels)):
                for node in nodes:
                    node.reason = "duplicate_label"
            result.append(ReplyNodes(owner, ordinal, prose, nodes))
        for handle, reply in zip(handles, result, strict=True):
            if reply.node is not handle:
                await handle.dispose()
        await self.check_origin()
        if self.registry and any(
            capability.signature != self.signature(result) for capability in self.registry.values()
        ):
            self.invalidate()
        return result

    @staticmethod
    def signature(replies: list[ReplyNodes]) -> tuple[Any, ...]:
        return tuple(
            (
                reply.ordinal,
                reply.prose,
                tuple(
                    (node.label, tuple(sorted(node.metadata.items())), node.reason)
                    for node in reply.options
                ),
            )
            for reply in replies
        )

    def publish(
        self,
        replies: list[ReplyNodes],
        ids: dict[int, str],
    ) -> dict[int, list[ReplyOption]]:
        self.invalidate()
        if not replies or replies[-1].ordinal not in ids:
            return {}
        reply = replies[-1]
        published: list[ReplyOption] = []
        signature = self.signature(replies)
        for node in reply.options:
            if node.reason is not None:
                continue
            identifier = secrets.token_urlsafe(24)
            self.registry[identifier] = Capability(
                reply, node, ids[reply.ordinal], self.generation, signature
            )
            published.append(ReplyOption(id=identifier, label=node.label))
        return {reply.ordinal: published}

    async def validate(self, identifier: str) -> Capability:
        capability = self.registry.get(identifier)
        if capability is None or capability.generation != self.generation:
            raise ChannelError("channel.select_option", "reply option is unknown or consumed")
        try:
            replies = await self.collect()
        except (ChannelError, PlaywrightError):
            self.invalidate()
            raise
        if self.signature(replies) != capability.signature or not replies:
            self.invalidate()
            raise ChannelError("channel.select_option", "reply option context changed")
        latest = replies[-1]
        same_owner = await latest.node.evaluate(
            "(node, owner) => node === owner", capability.reply.node
        )
        matches = [
            node
            for node in latest.options
            if node.reason is None
            and node.metadata["node_identity"] == capability.option.metadata["node_identity"]
        ]
        if not same_owner or len(matches) != 1:
            self.invalidate()
            raise ChannelError("channel.select_option", "reply option ownership changed")
        await self.check_origin()
        return capability

    async def inspect(self) -> list[dict[str, Any]]:
        replies = await self.collect()
        return [
            {
                "ordinal": reply.ordinal,
                "text": reply.prose,
                "prose": reply.prose,
                "latest": reply.ordinal == len(replies) - 1,
                "options": [
                    {
                        "label": node.label,
                        "eligible": node.reason is None and reply is replies[-1],
                        "reason": node.reason
                        or (None if reply is replies[-1] else "not_latest_reply"),
                        **node.metadata,
                    }
                    for node in reply.options
                ],
            }
            for reply in replies
        ]


async def inspect_reply_options(
    scope: Page | FrameLocator,
    config: ChannelConfig,
) -> list[dict[str, Any]]:
    """Inspect an existing native discovery scope, without navigation or activation.

    The caller owns frame resolution. Each wrapper entry contains prose and option
    labels, eligibility/reason, connected/visible/enabled and native node metadata.
    """
    if config.reply_options is None:
        return []
    # Lazy import avoids a cycle while sharing the channel's exact locator semantics.
    from syntheticcx.adapters.channel.playwright import locate

    inspector = ReplyOptions(scope, config, locate)
    try:
        await inspector.bind()
        return await inspector.inspect()
    finally:
        await inspector.dispose()
