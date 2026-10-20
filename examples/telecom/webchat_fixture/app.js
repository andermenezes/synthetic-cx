"use strict";

// The fixture deliberately describes a single supported widget, not arbitrary chat UI.
const params = new URLSearchParams(window.location.search);
const variants = new Set([
  "success", "repeated_question", "streamed_response", "missing_control", "silent_response"
]);
const echoes = new Set(["match", "normalized", "none", "mismatch", "mutating", "duplicate"]);
const variant = params.get("variant") ?? "success";
const view = params.get("view") ?? "main";
const echo = params.get("echo") ?? "match";
const greeting = params.get("greeting") ?? "normal";
const identity = params.get("identity") ?? "attribute";
const completionMode = params.get("completion") ?? "replace";

if (!variants.has(variant) || !["main", "frame"].includes(view) ||
    !echoes.has(echo) || !["normal", "slow"].includes(greeting) ||
    !["attribute", "none"].includes(identity) || !["replace", "reuse"].includes(completionMode)) {
  throw new Error("Unsupported fixture query parameter");
}

const page = document.body.dataset.page;
const widget = document.querySelector('[data-testid="chat-widget"]');
const messages = document.querySelector('[data-testid="chat-messages"]');
const status = document.querySelector('[data-testid="response-status"]');
const input = document.querySelector('[data-testid="chat-input"]');
const sendButton = document.querySelector('[data-testid="send-message"]');
let opened = false;
let submission = 0;
let pending = false;

if (identity === "none") {
  messages.querySelectorAll("[data-message-id]").forEach(node => node.removeAttribute("data-message-id"));
}

function bubble(actor, id, content) {
  const element = document.createElement("p");
  element.dataset.testid = `${actor}-message`;
  if (identity === "attribute") element.dataset.messageId = id;
  element.dataset.actor = actor;
  element.setAttribute("role", "article");
  element.setAttribute("aria-label", `${actor === "assistant" ? "Assistant" : "Customer"} message`);
  element.textContent = content;
  messages.append(element);
  return element;
}

function finish(responseId) {
  status.querySelector('[data-testid="streaming-indicator"]')?.remove();
  const marker = completionMode === "reuse" ?
    status.querySelector('[data-testid="response-complete"]') ?? document.createElement("span") :
    document.createElement("span");
  marker.dataset.testid = "response-complete";
  marker.dataset.responseId = responseId;
  marker.textContent = "Response complete";
  marker.hidden = false;
  status.append(marker);
  pending = false;
}

function activateChat() {
  if (opened) return;
  opened = true;
  widget.hidden = false;
  bubble("assistant", "greeting-1", "Hello! How can I help with your bill?");
  const indicator = document.createElement("span");
  indicator.dataset.testid = "streaming-indicator";
  indicator.dataset.responseId = "greeting-1";
  indicator.textContent = "Greeting in progress";
  status.append(indicator);
  window.setTimeout(() => finish("greeting-1"), greeting === "slow" ? 350 : 120);
}

function respond(responseId) {
  if (variant === "silent_response") return;
  if (variant === "repeated_question") {
    bubble("assistant", `${responseId}-1`, "Please give me your account number again.");
    finish(responseId);
    return;
  }
  if (variant === "streamed_response") {
    const indicator = document.createElement("span");
    indicator.dataset.testid = "streaming-indicator";
    indicator.dataset.responseId = responseId;
    indicator.textContent = "Reply in progress";
    status.append(indicator);
    const part = bubble("assistant", `${responseId}-1`, "Your bill");
    window.setTimeout(() => { part.textContent = "Your bill increased"; }, 40);
    window.setTimeout(() => {
      part.textContent = "Your bill increased because your plan changed.";
      bubble("assistant", `${responseId}-2`, "The new plan costs 50 more this month.");
      finish(responseId);
    }, 80);
    return;
  }
  // The success variant supplies two finalized assistant bubbles in DOM order.
  bubble("assistant", `${responseId}-1`, "Your bill increased because your plan changed.");
  bubble("assistant", `${responseId}-2`, "The new plan costs 50 more this month.");
  finish(responseId);
}

function submit() {
  if (!opened || pending || !input || !input.value.trim()) return;
  const content = input.value.trim();
  input.value = "";
  submission += 1;
  pending = true;
  const responseId = `response-${submission}`;
  const oldMarker = completionMode === "reuse" ?
    status.querySelector('[data-testid="response-complete"]') : null;
  status.replaceChildren(...(oldMarker ? [oldMarker] : []));
  if (oldMarker) oldMarker.hidden = true;
  if (echo !== "none") {
    const customer = bubble(
      "customer", `customer-${submission}`,
      echo === "match" ? content :
        echo === "normalized" ? `  ${content.toUpperCase().replace(/\s+/g, "  ")}  ` :
          ["mutating", "duplicate"].includes(echo) ? content :
          "A different customer message was displayed."
    );
    if (echo === "mutating") {
      window.setTimeout(() => { customer.textContent = "A different customer message was displayed."; }, 60);
    }
    if (echo === "duplicate") {
      window.setTimeout(() => bubble("customer", `customer-${submission}-extra`, content), 60);
    }
  }
  window.setTimeout(() => respond(responseId), completionMode === "reuse" ? 160 : 20);
}

if (variant === "missing_control") {
  input.remove();
}

sendButton.addEventListener("click", submit);
input?.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.isComposing) {
    event.preventDefault();
    submit();
  }
});

if (page === "frame") {
  // The parent opens an existing frame; it never replaces the document at activation.
  window.activateChat = activateChat;
} else {
  const frame = document.querySelector('[data-testid="chat-frame"]');
  const opener = document.querySelector('[data-testid="open-chat"]');
  if (view === "frame") {
    widget.hidden = true;
    const frameUrl = new URL("frame.html", window.location.href);
    frameUrl.search = new URLSearchParams({ variant, echo, greeting, identity, completion: completionMode }).toString();
    const frameReady = () => {
      // The initial frame.html load may complete after the configured navigation starts.
      if (frame.contentWindow.location.search !== frameUrl.search ||
          typeof frame.contentWindow.activateChat !== "function") return;
      opener.disabled = false;
      frame.removeEventListener("load", frameReady);
    };
    frame.addEventListener("load", frameReady);
    frame.src = frameUrl.href;
  } else {
    opener.disabled = false;
  }
  opener.addEventListener("click", () => {
    if (view === "frame") {
      frame.hidden = false;
      frame.contentWindow.activateChat();
    } else {
      activateChat();
    }
  });
}
