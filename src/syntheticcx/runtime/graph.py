"""Single-run LangGraph conversation routing over owned protocols and evidence."""

from __future__ import annotations

from typing import Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from syntheticcx.config.models import ExecutionLimits
from syntheticcx.domain.events import (
    ConversationStarted,
    MessageObserved,
    RepeatedInformationObserved,
    RunLifecycleChange,
)
from syntheticcx.domain.models import (
    ConversationObservation,
    CustomerScenarioContext,
    Message,
    PersonaDefinition,
    PersonaState,
    PersonaStateTransition,
    Scenario,
)
from syntheticcx.domain.protocols import ChannelAdapter, PersonaSimulator, WaitController
from syntheticcx.errors import ChannelError, ModelError
from syntheticcx.runtime.behavior import apply_behavior
from syntheticcx.runtime.journal import RunJournal
from syntheticcx.runtime.repetition import RepetitionTracker
from syntheticcx.runtime.state import RuntimeProgress, RuntimeState


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
) -> CompiledStateGraph[RuntimeState, None, RuntimeState, RuntimeState]:
    """Build one graph per run; its repetition tracker is deliberately not shared."""
    tracker = RepetitionTracker(scenario.information)

    async def initialize(state: RuntimeState) -> dict[str, object]:
        lifecycle = await journal.append(RunLifecycleChange(previous=None, current="in_progress"))
        try:
            initialized = await wait.run(
                simulator.initialize(persona, CustomerScenarioContext.from_scenario(scenario)),
                limits.model_timeout_seconds,
            )
        except Exception:
            raise ModelError("persona.initialize", "persona initialization failed") from None
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
        response = None
        if limits.greeting_grace_seconds > 0:
            try:
                response = await wait.run(channel.receive(), limits.greeting_grace_seconds)
            except TimeoutError:
                pass
            except Exception:
                raise ChannelError("channel.greeting", "greeting observation failed") from None
        if response is not None:
            for message in response.messages:
                await journal.append(MessageObserved(message=message))
        await journal.append(
            ConversationStarted(
                greeting_observed=response is not None,
                grace_seconds=limits.greeting_grace_seconds,
                greeting_message_ids=[message.id for message in response.messages]
                if response
                else [],
            )
        )
        observation = (
            ConversationObservation(kind="assistant_response", messages=response.messages)
            if response
            else ConversationObservation(kind="conversation_started")
        )
        return {"observation": observation, "response": response}

    async def behavior(state: RuntimeState) -> dict[str, object]:
        response = state["response"]
        current = state["persona_state"]
        if response is not None:
            for repetition in tracker.detect(response):
                event = await journal.append(
                    RepeatedInformationObserved(
                        information_id=repetition.information_id,
                        original_message_id=repetition.original_message_id,
                        request_message_id=repetition.request_message_id,
                    )
                )
                transition = apply_behavior(
                    current,
                    persona.behavioral_rules,
                    event="repeated_information",
                    trigger_ids=[
                        event.id,
                        repetition.original_message_id,
                        repetition.request_message_id,
                    ],
                )
                await journal.append(transition)
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
        try:
            action = await wait.run(
                simulator.respond(observation, state["persona_state"]), limits.model_timeout_seconds
            )
        except Exception:
            raise ModelError("persona.respond", "persona response failed") from None
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
        elif action.message is None:
            raise ModelError("persona.respond", "continuing action has no customer message")
        return {
            "persona_state": current,
            "action": action,
            "pending_message": action.message if reason is None else None,
            "termination_reason": reason,
        }

    async def send(state: RuntimeState) -> dict[str, object]:
        content = state["pending_message"]
        if content is None or state["termination_reason"] is not None:
            raise ValueError("cannot send after termination or without a message")
        try:
            receipt = await wait.run(channel.send(content), limits.browser_action_timeout_seconds)
        except Exception:
            raise ChannelError("channel.send", "customer send failed") from None
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
                ),
                identity_evidence=receipt.evidence,
            )
        )
        await journal.append(
            PersonaStateTransition(
                trigger_ids=[observed.id],
                before=before,
                after=after,
                applied_rule_ids=[],
            )
        )
        tracker.record_send(content, receipt)
        return {
            "receipt": receipt,
            "turn_count": state["turn_count"] + 1,
            "persona_state": after,
            "pending_message": None,
        }

    async def receive(state: RuntimeState) -> dict[str, object]:
        try:
            response = await wait.run(channel.receive(), limits.response_timeout_seconds)
        except Exception:
            raise ChannelError("channel.receive", "assistant receive failed") from None
        for message in response.messages:
            await journal.append(MessageObserved(message=message))
        return {
            "response": response,
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
