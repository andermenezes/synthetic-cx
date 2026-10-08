"""Owned bounded DOM helpers shared by discovery inspections, never model-authored JS."""

DOM_HELPERS = """
const walkDOM = (root, shadows = true, text = false) => {
  const nodes = [];
  const stack = [{next: root.firstChild, depth: 0}];
  let visited = 0, roots = 0;
  if (shadows && root.shadowRoot) {
    roots = 1;
    stack.push({next: root.shadowRoot.firstChild, depth: 1});
  }
  while (stack.length) {
    const frame = stack[stack.length - 1], node = frame.next;
    if (!node) { stack.pop(); continue; }
    frame.next = node.nextSibling;
    if (++visited > 20000) return {nodes, complete: false};
    if (node.nodeType === 1 || (text && node.nodeType === 3)) nodes.push(node);
    if (node.firstChild) stack.push({next: node.firstChild, depth: frame.depth});
    if (shadows && node.shadowRoot) {
      if (++roots > 128 || frame.depth >= 32) return {nodes, complete: false};
      stack.push({next: node.shadowRoot.firstChild, depth: frame.depth + 1});
    }
  }
  return {nodes, complete: true};
};
const composedParent = node => node.parentElement || node.getRootNode().host || null;
const composedClosest = (node, selector) => {
  for (let i = 0; node && i < 20000; i++, node = composedParent(node))
    if (node.matches(selector)) return node;
  return null;
};
const composedContains = (outer, inner) => {
  for (let i = 0; inner && i < 20000; i++, inner = composedParent(inner))
    if (inner === outer) return true;
  return false;
};
const renderedElement = node => {
  if (!node.getClientRects().length) return false;
  for (let i = 0; node && i < 20000; i++, node = composedParent(node)) {
    const style = getComputedStyle(node);
    if (style.display === 'none' || style.visibility !== 'visible' ||
      style.contentVisibility === 'hidden' || style.opacity === '0') return false;
  }
  return true;
};
const scopeAddress = (node, doc, depth = 0) => {
  if (depth > 64) return null;
  if (node.host) {
    const prefix = scopeAddress(node.host, doc, depth + 1);
    return prefix && [...prefix, {shadow: true}];
  }
  const tree = node.getRootNode(), base = tree === doc ? doc.body : tree;
  if (node === doc.body) return [];
  const prefix = tree === doc ? [] : scopeAddress(tree, doc, depth + 1);
  if (!prefix) return null;
  const path = [];
  let current = node;
  while (current !== base) {
    const parent = current.parentNode;
    if (!parent || !parent.children || path.length >= 256) return null;
    const index = Array.prototype.indexOf.call(parent.children, current);
    if (index < 0) return null;
    path.unshift(index); current = parent;
  }
  return [...prefix, {path, id: node.id.length <= 160 ? node.id : '',
    tag: node.tagName, role: node.getAttribute('role') || ''}];
};
const resolveScope = (doc, address, expected = null, trace = null) => {
  if (!Array.isArray(address) || address.length > 65) return null;
  let root = doc.body;
  if (!address.length && expected && expected[0] !== root) return null;
  if (trace) trace.push(root);
  for (let index = 0; index < address.length; index++) {
    const step = address[index];
    if (step.shadow) {
      if (!root || !root.shadowRoot) return null;
      root = root.shadowRoot;
    } else {
      let node;
      if (step.id) {
        const scan = walkDOM(root, false);
        if (!scan.complete) return null;
        const matches = scan.nodes.filter(el => el.id === step.id);
        if (matches.length !== 1) return null;
        node = matches[0];
      } else {
        node = root;
        for (const index of step.path) node = node && node.children[index];
      }
      if (!node || node.tagName !== step.tag ||
          (node.getAttribute('role') || '') !== step.role) return null;
      root = node;
    }
    // Stable IDs remain authoritative; anonymous positional steps must stay the same node.
    if (!step.shadow && !step.id && expected && expected[index + 1] !== root) return null;
    if (trace) trace.push(root);
  }
  return root;
};
"""

TEXT_HELPER = """
const composedText = (node, limit, skipControls = false) => {
  if (!renderedElement(node)) return '';
  if (!walkDOM(node).complete) return null;
  const stack = [(node.shadowRoot ? node.shadowRoot.childNodes : node.childNodes).values()];
  const seen = new Set();
  let result = '';
  while (stack.length) {
    const next = stack[stack.length - 1].next();
    if (next.done) { stack.pop(); continue; }
    const child = next.value;
    if (seen.has(child)) continue;
    seen.add(child);
    if (seen.size > 20000) return null;
    if (child.nodeType === 3) result += child.textContent;
    else if (child.nodeType === 1) {
      if (child.matches('script, style, template, noscript')) continue;
      if (skipControls && child.matches('button, a, [role=button]')) continue;
      if (child.matches('input, textarea, select, [role=textbox], ' +
        '[contenteditable]:not([contenteditable=false])')) return null;
      const style = getComputedStyle(child);
      if (style.display === 'none' || style.visibility !== 'visible' ||
        style.contentVisibility === 'hidden' || style.opacity === '0') continue;
      const children = child.tagName === 'SLOT' ? child.assignedNodes({flatten: true}) : [];
      const collection = children.length ? children : child.shadowRoot ?
        child.shadowRoot.childNodes : child.childNodes;
      if (collection.length > 20000) return null;
      stack.push(collection.values());
    }
    if (result.length > limit) return result.slice(0, limit);
  }
  return result;
};
"""
