"""Single-run LangGraph conversation routing over owned protocols and evidence."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, cast, runtime_checkable

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from syntheticcx.config.models import BehaviorConfig, ExecutionLimits, JsonObject
from syntheticcx.domain.behavior import BehavioralObservation, BehavioralRepetitionFact
from syntheticcx.domain.events import (
    BehavioralObservationRecorded,
    BehavioralSignalRecorded,
    ConversationStarted,
    MessageObserved,
    MessageObservedEvent,
    PersonaOutputAmbiguityObserved,
    RepeatedInformationObserved,
    RunLifecycleChange,
)
from syntheticcx.domain.models import (
    BehavioralStateTransition,
    ConversationCapabilities,
    ConversationObservation,
    CustomerScenarioContext,
    Message,
    ModelInteraction,
    PersonaDefinition,
    PersonaState,
    PersonaStateTransition,
    Scenario,
    SelectionReceipt,
)
from syntheticcx.domain.protocols import (
    ChannelAdapter,
    ControlsObservingChannel,
    InitialObservingChannel,
    OptionSelectingChannel,
    PersonaSimulator,
    WaitController,
)
from syntheticcx.errors import BehaviorError, ModelError, safe_channel_error
from syntheticcx.runtime.behavior import apply_behavior
from syntheticcx.runtime.behavior_signals import BehaviorDetectors, prepare_behavior
from syntheticcx.runtime.journal import RunJournal
from syntheticcx.runtime.repetition import RepetitionTracker
from syntheticcx.runtime.state import RuntimeProgress, RuntimeState
from syntheticcx.telemetry.run import span


@runtime_checkable
class _InteractionSource(Protocol):
    def take_interactions(self) -> tuple[ModelInteraction, ...]: ...


def build_graph(
    *,
    scenario: Scenario,
    persona: PersonaDefinition,
    channel: ChannelAdapter,
    simulator: PersonaSimulator,
    journal: RunJournal,
    wait: WaitController,
    limits: ExecutionLimits,
    progress: RuntimeProgress | None = None,
    behavior_config: BehaviorConfig | None = None,
    behavior_detectors: BehaviorDetectors | None = None,
    remaining_time: Callable[[], float] | None = None,
) -> CompiledStateGraph[RuntimeState, None, RuntimeState, RuntimeState]:
    """Build one graph per run; its repetition tracker is deliberately not shared."""
    tracker = RepetitionTracker(scenario.information)
    detectors = (
        behavior_detectors
        or prepare_behavior(
            behavior_config if behavior_config is not None else BehaviorConfig(),
            persona.behavioral_rules,
        ).instantiate()
    )
    remaining = remaining_time or (lambda: limits.journey_timeout_seconds)
    occurrences: dict[str, int] = {}
    fired_rule_ids: set[str] = set()
    boundary_event_ids: list[str] = []
    last_customer_message: Message | None = None

    async def commit_interactions(input_ids: list[str]) -> None:
        """Acknowledge model evidence before any action depending on its output."""
        if not isinstance(simulator, _InteractionSource):
            return
        try:
            interactions = simulator.take_interactions()
        except Exception:
            raise ModelError(
                "persona.interactions", "persona interaction evidence failed"
            ) from None
        for interaction in interactions:
            omissions = list(interaction.capture_omissions)
            if not input_ids:
                omissions.append("input_event_unavailable")
            # The customer send or terminal action is not yet durable here. Do
            # not manufacture references to an event that may never be written.
            omissions.append("output_event_not_yet_committed")
            await journal.append(
                interaction.model_copy(
                    update={
                        "input_event_ids": input_ids,
                        "output_event_ids": [],
                        "capture_omissions": omissions,
                    },
                    deep=True,
                )
            )

    async def initialize(state: RuntimeState) -> dict[str, object]:
        lifecycle = await journal.append(RunLifecycleChange(previous=None, current="in_progress"))
        try:
            with span("persona.initialize"), span("model.call", operation="persona.initialize"):
                initialized = await wait.run(
                    simulator.initialize(persona, CustomerScenarioContext.from_scenario(scenario)),
                    limits.model_timeout_seconds,
                )
        except Exception:
            raise ModelError("persona.initialize", "persona initialization failed") from None
        await commit_interactions([lifecycle.id])
        if initialized != state["persona_state"]:
            await journal.append(
                PersonaStateTransition(
                    trigger_ids=[lifecycle.id],
                    before=state["persona_state"],
                    after=initialized,
                    applied_rule_ids=[],
                )
            )
        if progress is not None:
            progress.persona_state = initialized
        return {"persona_state": initialized}

    async def greeting(state: RuntimeState) -> dict[str, object]:
        nonlocal boundary_event_ids
        boundary_event_ids = []
        response = None
        if isinstance(channel, InitialObservingChannel):
            try:
                response = await wait.run(
                    channel.observe_initial(), limits.browser_action_timeout_seconds
                )
            except Exception as exc:
                raise safe_channel_error(
                    exc, "channel.greeting", "initial chat observation failed"
                ) from None
        if (
            not isinstance(channel, InitialObservingChannel)
            and response is None
            and limits.greeting_grace_seconds > 0
        ):
            try:
                response = await wait.run(channel.receive(), limits.greeting_grace_seconds)
            except TimeoutError:
                pass
            except Exception as exc:
                raise safe_channel_error(
                    exc, "channel.greeting", "greeting observation failed"
                ) from None
        if response is not None:
            messages: list[Message] = []
            raw_messages: list[Message] = []
            for message in response.messages:
                committed = await journal.append(MessageObserved(message=message))
                assert isinstance(committed, MessageObservedEvent)
                boundary_event_ids.append(committed.id)
                messages.append(committed.data.message)
                raw_messages.append(
                    message.model_copy(
                        update={
                            "id": committed.data.message.id,
                            "sequence": committed.data.message.sequence,
                        },
                        deep=True,
                    )
                )
            raw_response = response.model_copy(update={"messages": raw_messages}, deep=True)
            response = response.model_copy(update={"messages": messages})
        started = await journal.append(
            ConversationStarted(
                greeting_observed=response is not None,
                grace_seconds=limits.greeting_grace_seconds,
                greeting_message_ids=[message.id for message in response.messages]
                if response
                else [],
            )
        )
        boundary_event_ids.append(started.id)
        observation = (
            ConversationObservation(kind="assistant_response", messages=response.messages)
            if response
            else ConversationObservation(kind="conversation_started")
        )
        return {"observation": observation, "response": raw_response if response else None}

    async def behavior(state: RuntimeState) -> dict[str, object]:
        response = state["response"]
        current = state["persona_state"]
        facts: list[BehavioralRepetitionFact] = []
        if response is not None:
            for repetition in tracker.detect(response):
                event = await journal.append(
                    RepeatedInformationObserved(
                        information_id=repetition.information_id,
                        original_message_id=repetition.original_message_id,
                        request_message_id=repetition.request_message_id,
                    )
                )
                facts.append(
                    BehavioralRepetitionFact(
                        event_id=event.id,
                        information_id=repetition.information_id,
                        original_message_id=repetition.original_message_id,
                        request_message_id=repetition.request_message_id,
                    )
                )
        observation = BehavioralObservation(
            kind="greeting" if state["turn_count"] == 0 else "customer_turn_completed",
            messages=tuple(response.messages) if response is not None else (),
            evidence_event_ids=tuple(boundary_event_ids) + tuple(fact.event_id for fact in facts),
            state=current.model_copy(deep=True),
            turn_count=state["turn_count"],
            last_customer_message=last_customer_message,
            repetition_facts=tuple(facts),
        )
        committed_events = journal.committed_events
        signals = await detectors.detect(
            observation,
            event_ids=frozenset(event.id for event in committed_events),
            message_ids=frozenset(
                event.data.message.id
                for event in committed_events
                if isinstance(event, MessageObservedEvent)
            ),
            wait=wait,
            timeout_seconds=limits.model_timeout_seconds,
            remaining_time=remaining,
        )
        # Validate the complete redacted batch before its boundary is acknowledged.
        safe_signals = []
        try:
            for signal in signals:
                safe_signal = journal.sanitize_evidence(signal)
                detectors.validate_sanitized(signal, safe_signal)
                safe_signals.append(safe_signal)
        except Exception:
            raise BehaviorError(
                "behavior.sanitize", "behavior evidence cannot be safely persisted"
            ) from None
        for interaction in detectors.take_interactions():
            await journal.append(interaction)
        await journal.append(
            BehavioralObservationRecorded(
                observation_id=observation.observation_id,
                kind=observation.kind,
                evidence_event_ids=list(observation.evidence_event_ids),
                message_ids=[message.id for message in observation.messages],
                detector_ids=[metadata.detector_id for metadata in detectors.metadata],
                state=current,
                turn_count=state["turn_count"],
                signal_count=len(safe_signals),
            )
        )
        signal_events = [
            await journal.append(BehavioralSignalRecorded(signal=signal)) for signal in safe_signals
        ]
        for signal, event in zip(safe_signals, signal_events, strict=True):
            occurrences[signal.signal] = occurrences.get(signal.signal, 0) + 1
            transition = apply_behavior(
                current,
                persona.behavioral_rules,
                signal=signal.signal,
                trigger_ids=[event.id],
                occurrence=occurrences[signal.signal],
                fired_rule_ids=fired_rule_ids,
            )
            await journal.append(
                BehavioralStateTransition(
                    observation_id=observation.observation_id,
                    signal_event_id=event.id,
                    signal=signal.signal,
                    occurrence=occurrences[signal.signal],
                    before=transition.before,
                    after=transition.after,
                    applied_rule_ids=transition.applied_rule_ids,
                )
            )
            fired_rule_ids.update(transition.applied_rule_ids)
            current = transition.after
            if progress is not None:
                progress.persona_state = current
        return {
            "persona_state": current,
            "termination_reason": (
                "abandonment"
                if current.abandoned
                else "max_turns"
                if state["turn_count"] >= limits.max_turns
                else state["termination_reason"]
            ),
        }

    async def respond(state: RuntimeState) -> dict[str, object]:
        observation = state["observation"]
        assert observation is not None
        if isinstance(channel, ControlsObservingChannel):
            try:
                controls = await wait.run(
                    channel.observe_controls(), limits.browser_action_timeout_seconds
                )
                capabilities = ConversationCapabilities.model_validate(controls)
                capabilities = journal.sanitize_evidence(capabilities)
            except Exception as exc:
                raise safe_channel_error(
                    exc, "channel.controls", "current controls observation failed"
                ) from None
            observation = observation.model_copy(update={"capabilities": capabilities})
        try:
            with span("persona.respond"), span("model.call", operation="persona.respond"):
                action = await wait.run(
                    simulator.respond(observation, state["persona_state"]),
                    limits.model_timeout_seconds,
                )
        except Exception as exc:
            if isinstance(exc, ModelError):
                raise
            raise ModelError(
                "persona.respond",
                "persona response failed",
                reason_code=(
                    "persona_timeout" if isinstance(exc, TimeoutError) else "unknown_model_failure"
                ),
            ) from None
        if action.discarded_action_count:
            await journal.append(
                PersonaOutputAmbiguityObserved(
                    candidate_action_count=action.discarded_action_count + 1,
                    selected_action_order=1,
                    discarded_action_count=action.discarded_action_count,
                )
            )
        observation_ids = {message.id for message in observation.messages}
        input_ids = [
            event.id
            for event in journal.committed_events
            if (event.type == "message_observed" and event.data.message.id in observation_ids)
            or (observation.kind == "conversation_started" and event.type == "conversation_started")
        ]
        await commit_interactions(input_ids)
        current: PersonaState = state["persona_state"]
        for transition in action.transitions:
            if transition.before != current:
                raise ModelError("persona.respond", "persona transition is inconsistent")
            await journal.append(transition)
            current = transition.after
            if progress is not None:
                progress.persona_state = current
        if current != action.state:
            raise ModelError("persona.respond", "persona action lacks a state transition")
        reason: str | None = None
        if current.abandoned or action.signal == "abandon":
            reason = "abandonment"
        elif action.signal == "goal_believed_complete":
            reason = "goal_believed_complete"
        elif (action.message is None) == (action.selected_option_id is None):
            raise ModelError("persona.respond", "continuing action requires exactly one decision")
        if reason is None and action.selected_option_id is not None:
            if sum(o.id == action.selected_option_id for o in observation.current_options) != 1:
                raise ModelError("persona.respond", "persona selected an unavailable option")
        if reason is None and action.message is not None and not observation.text_available:
            raise ModelError("persona.respond", "persona text action is unavailable")
        return {
            "observation": observation,
            "persona_state": current,
            "action": action,
            "pending_message": action.message if reason is None else None,
            "termination_reason": reason,
        }

    async def send(state: RuntimeState) -> dict[str, object]:
        nonlocal last_customer_message
        action = state["action"]
        if action is None or state["termination_reason"] is not None:
            raise ValueError("cannot send after termination or without an action")
        metadata: JsonObject = {}
        selection = action.selected_option_id is not None
        try:
            if selection:
                option_id = action.selected_option_id
                assert option_id is not None
                observation = state["observation"]
                latest = observation.messages[-1] if observation and observation.messages else None
                if (
                    observation is None
                    or sum(o.id == option_id for o in observation.current_options) != 1
                ):
                    raise ValueError("unavailable option")
                if not isinstance(channel, OptionSelectingChannel):
                    raise ValueError("channel does not support option selection")
                receipt = await wait.run(
                    channel.select_option(option_id), limits.browser_action_timeout_seconds
                )
                if not isinstance(receipt, SelectionReceipt):
                    raise ValueError("selection receipt does not match the requested option")
                # The observation is committed/redacted; adapter receipt IDs may
                # still contain raw DOM attributes. Compare in its owned namespace.
                safe_receipt = journal.sanitize_evidence(receipt)
                if safe_receipt.selected_option_id != option_id or (
                    observation.capabilities is None
                    and (latest is None or safe_receipt.source_message_id != latest.id)
                ):
                    raise ValueError("selection receipt does not match the requested option")
                content = receipt.selected_label
                metadata = {
                    "action_kind": "select_option",
                    "option_id": receipt.selected_option_id,
                    "source_assistant_message_id": receipt.source_message_id,
                }
            else:
                pending = state["pending_message"]
                if pending is None:
                    raise ValueError("cannot send without a message")
                content = pending
                receipt = await wait.run(
                    channel.send(content), limits.browser_action_timeout_seconds
                )
        except Exception as exc:
            # Both decisions use the existing send operation/runner error boundary.
            message = "customer action failed" if selection else "customer send failed"
            raise safe_channel_error(exc, "channel.send", message) from None
        before = state["persona_state"]
        after = before.model_copy(update={"turn_number": state["turn_count"] + 1})
        if progress is not None:
            progress.turn_count += 1
            progress.persona_state = after
        # The receipt identifies the actual submitted customer message. No echo is
        # needed to make the submitted content canonical evidence.
        observed = await journal.append(
            MessageObserved(
                message=Message(
                    id=receipt.message_id,
                    sequence=state["turn_count"],
                    actor="customer",
                    content=content,
                    observed_at=receipt.submitted_at,
                    metadata=metadata,
                ),
                identity_evidence=receipt.evidence,
            )
        )
        with span("persona.update"):
            await journal.append(
                PersonaStateTransition(
                    trigger_ids=[observed.id],
                    before=before,
                    after=after,
                    applied_rule_ids=[],
                )
            )
        if not selection:
            assert isinstance(observed, MessageObservedEvent)
            tracker.record_send(
                content,
                receipt.model_copy(update={"message_id": observed.data.message.id}, deep=True),
            )
            last_customer_message = observed.data.message.model_copy(
                update={"content": content}, deep=True
            )
        else:
            last_customer_message = None
        return {
            "receipt": receipt,
            "turn_count": state["turn_count"] + 1,
            "persona_state": after,
            "pending_message": None,
        }

    async def receive(state: RuntimeState) -> dict[str, object]:
        nonlocal boundary_event_ids
        try:
            with span("conversation.turn", **{"turn.number": state["turn_count"]}):
                response = await wait.run(channel.receive(), limits.response_timeout_seconds)
        except Exception as exc:
            raise safe_channel_error(exc, "channel.receive", "assistant receive failed") from None
        messages: list[Message] = []
        raw_messages: list[Message] = []
        boundary_event_ids = []
        for message in response.messages:
            committed = await journal.append(MessageObserved(message=message))
            assert isinstance(committed, MessageObservedEvent)
            boundary_event_ids.append(committed.id)
            messages.append(committed.data.message)
            raw_messages.append(
                message.model_copy(
                    update={
                        "id": committed.data.message.id,
                        "sequence": committed.data.message.sequence,
                    },
                    deep=True,
                )
            )
        raw_response = response.model_copy(update={"messages": raw_messages}, deep=True)
        response = response.model_copy(update={"messages": messages})
        return {
            "response": raw_response,
            "observation": ConversationObservation(
                kind="assistant_response", messages=response.messages
            ),
        }

    def after_response(state: RuntimeState) -> Literal["send", "terminal"]:
        return "terminal" if state["termination_reason"] is not None else "send"

    def after_behavior(state: RuntimeState) -> Literal["respond", "terminal"]:
        return "terminal" if state["termination_reason"] is not None else "respond"

    graph = StateGraph(RuntimeState)
    graph.add_node("initialize", initialize)
    graph.add_node("greeting", greeting)
    graph.add_node("behavior", behavior)
    graph.add_node("respond", respond)
    graph.add_node("send", send)
    graph.add_node("receive", receive)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "greeting")
    graph.add_edge("greeting", "behavior")
    graph.add_conditional_edges("behavior", after_behavior, {"respond": "respond", "terminal": END})
    graph.add_conditional_edges("respond", after_response, {"send": "send", "terminal": END})
    graph.add_edge("send", "receive")
    graph.add_edge("receive", "behavior")
    return cast(
        "CompiledStateGraph[RuntimeState, None, RuntimeState, RuntimeState]", graph.compile()
    )
