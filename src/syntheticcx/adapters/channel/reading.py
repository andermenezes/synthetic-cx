"""Read-only composed-DOM prose extraction for option-enabled channels."""

from syntheticcx.adapters.discovery.dom import DOM_HELPERS, TEXT_HELPER

# Omit *all* controls, including unsafe/unconfigured ones. Their labels must never
# become finalized prose and later mutate the transcript when controls disappear.
READING_HELPERS = (
    DOM_HELPERS
    + TEXT_HELPER
    + """
const controls = 'button, a, input, textarea, select, [role=button], [role=link], ' +
  '[role=textbox], [contenteditable]:not([contenteditable=false])';
const proseText = root => {
  if (root.matches('script,style,template,noscript,input,textarea,select,' +
    '[role=textbox],[contenteditable]:not([contenteditable=false])')) return '';
  if (!walkDOM(root).complete) throw new Error('incomplete prose scan');
  if (!renderedElement(root)) return '';
  const stack = [(root.shadowRoot ? root.shadowRoot.childNodes : root.childNodes).values()];
  const seen = new Set();
  let result = '';
  while (stack.length) {
    const next = stack[stack.length - 1].next();
    if (next.done) { stack.pop(); continue; }
    const node = next.value;
    if (seen.has(node)) continue;
    seen.add(node);
    if (seen.size > 20000) throw new Error('incomplete prose scan');
    if (node.nodeType === 3) result += node.textContent;
    else if (node.nodeType === 1) {
      if (node.matches(controls + ', script, style, template, noscript, time')) continue;
      const style = getComputedStyle(node);
      if (style.display === 'none' || style.visibility !== 'visible' ||
          style.contentVisibility === 'hidden' || style.opacity === '0') continue;
      const assigned = node.tagName === 'SLOT' ? node.assignedNodes({flatten: true}) : [];
      const children = assigned.length ? assigned : node.shadowRoot ?
        node.shadowRoot.childNodes : node.childNodes;
      stack.push(children.values());
    }
    if (result.length > 100000) throw new Error('prose size limit exceeded');
  }
  return result.replace(/\\s+/g, ' ').trim();
};
// Explicit accessible-name references may be hidden. Editors and executable
// text are still excluded; values never become observed control labels.
const labelText = root => {
  const stack = [root];
  const seen = new Set();
  let result = '';
  while (stack.length) {
    const node = stack.pop();
    if (seen.has(node)) continue;
    seen.add(node);
    if (seen.size > 20000) throw new Error('incomplete label scan');
    if (node.nodeType === 3) result += node.textContent;
    else if (node.nodeType === 1) {
      if (node.matches('input,textarea,select,[role=textbox],[contenteditable=true],' +
        'script,style,template,noscript')) continue;
      const assigned = node.tagName === 'SLOT' ? node.assignedNodes({flatten:true}) : [];
      const children = assigned.length ? assigned : node.shadowRoot ?
        node.shadowRoot.childNodes : node.childNodes;
      stack.push(...[...children].reverse());
    }
    if (result.length > 1000) throw new Error('label size limit exceeded');
  }
  return result.replace(/\\s+/g, ' ').trim();
};
const assistantMarkers = new Set(['assistant','bot','agent','assistente','received','inbound']);
const customerMarkers = new Set(['customer','user','client','usuario','cliente','sent','outgoing']);
const markerKinds = value => {
  const tokens = value.toLowerCase().split(/[\\s_-]+/);
  return [tokens.some(token => assistantMarkers.has(token)),
    tokens.some(token => customerMarkers.has(token))];
};
const actorSignals = node => {
  const explicit = ['data-role','data-actor','data-author','data-sender']
    .map(name => (node.getAttribute(name) || '').toLowerCase());
  let assistant = explicit.some(value => assistantMarkers.has(value));
  let customer = explicit.some(value => customerMarkers.has(value));
  if (!assistant && !customer) [assistant, customer] = markerKinds(
    String(node.className) + ' ' + (node.getAttribute('aria-label') || ''));
  return [assistant, customer];
};
const observedActor = node => {
  const [assistant, customer] = actorSignals(node);
  return assistant === customer ? null : assistant ? 'assistant' : 'customer';
};
const observedActorConflict = node => actorSignals(node).every(Boolean);
const observedActorSelector = node => {
  const tag = node.tagName.toLowerCase();
  for (const attribute of ['data-role','data-actor','data-author','data-sender']) {
    const value = node.getAttribute(attribute);
    if (value && (assistantMarkers.has(value.toLowerCase()) ||
      customerMarkers.has(value.toLowerCase())))
      return tag + '[' + attribute + '=' + CSS.escape(value) + ']';
  }
  const testId = node.getAttribute('data-testid');
  if (testId) return tag + '[data-testid=' + CSS.escape(testId) + ']';
  const marked = [...node.classList].filter(value => markerKinds(value).some(Boolean));
  if (marked.length) return tag + marked.map(value => '.' + CSS.escape(value)).join('');
  return node.classList.length ? tag +
    [...node.classList].map(value => '.' + CSS.escape(value)).join('') :
    node.id ? '#' + CSS.escape(node.id) : tag;
};
const safeComposerElement = node => {
  if (!node.isConnected || !renderedElement(node) || node.matches(':disabled,[readonly]') ||
    (node.getAttribute('aria-readonly') || '').toLowerCase() === 'true') return false;
  if (node.tagName === 'INPUT' && node.type !== 'text') return false;
  const purpose = ['id','name','aria-label','placeholder'].map(key => node.getAttribute(key) || '')
    .join(' ').replace(/([a-z])([A-Z])/g, '$1 $2').toLowerCase()
    .replace(/[^\\p{L}\\p{N}]+/gu, ' ').trim();
  const forbidden = ['login','log in','sign in','password','passwd','otp','one time code',
    'verification code','credit card','card number','payment','purchase','consent','upload',
    'senha','pagamento','comprar','cartao','cartão','aceitar'];
  if (forbidden.some(term => (' ' + purpose + ' ').includes(' ' + term + ' '))) return false;
  return node.matches('textarea,input:not([type]),input[type=text],' +
    '[role=textbox],[contenteditable=true]');
};
"""
)
