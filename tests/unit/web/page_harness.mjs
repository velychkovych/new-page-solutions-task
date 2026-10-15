// Runs the real <script> blocks of web/index.html in a Node vm context with a minimal
// DOM stand-in and a scripted fetch, so the page's own JavaScript is what the tests
// exercise. Driven by test_page_behaviour.py: a JSON scenario on stdin, a JSON report on
// stdout. Only the DOM surface index.html uses is implemented; an unknown call throws,
// which fails the test loudly instead of passing silently.
import { readFileSync } from "node:fs";
import vm from "node:vm";

const [indexPath] = process.argv.slice(2);
const html = readFileSync(indexPath, "utf8");
const scenario = JSON.parse(readFileSync(0, "utf8"));

function scriptBlock(id) {
  const match = html.match(new RegExp(`<script id="${id}">([\\s\\S]*?)</script>`));
  if (!match) throw new Error(`no <script id="${id}"> in index.html`);
  return match[1];
}

// --- DOM stand-in -----------------------------------------------------------------

class FakeText {
  constructor(data) { this.data = String(data); this.parent = null; }
  get textContent() { return this.data; }
}

class FakeElement {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.attributes = new Map();
    this.children = [];
    this.parent = null;
    this.hidden = false;
    this.disabled = false;
    this.value = "";
    this.dataset = {};
    this.listeners = new Map();
    const self = this;
    this.classList = {
      add(name) { const s = self._classes(); s.add(name); self._setClasses(s); },
      remove(name) { const s = self._classes(); s.delete(name); self._setClasses(s); },
      contains(name) { return self._classes().has(name); },
    };
  }
  _classes() { return new Set((this.attributes.get("class") || "").split(/\s+/).filter(Boolean)); }
  _setClasses(set) { this.attributes.set("class", [...set].join(" ")); }
  setAttribute(key, value) { this.attributes.set(key, String(value)); }
  getAttribute(key) { return this.attributes.has(key) ? this.attributes.get(key) : null; }
  _adopt(nodes) {
    return nodes.map((n) => {
      const node = typeof n === "string" ? new FakeText(n) : n;
      if (!(node instanceof FakeElement || node instanceof FakeText)) throw new Error(`cannot append ${n}`);
      node.parent?._detach(node);
      node.parent = this;
      return node;
    });
  }
  _detach(node) { this.children = this.children.filter((c) => c !== node); node.parent = null; }
  append(...nodes) { this.children.push(...this._adopt(nodes)); }
  prepend(...nodes) { this.children.unshift(...this._adopt(nodes)); }
  replaceChildren(...nodes) {
    for (const c of this.children) c.parent = null;
    this.children = [];
    this.append(...nodes);
  }
  remove() { this.parent?._detach(this); }
  get textContent() { return this.children.map((c) => c.textContent).join(""); }
  set textContent(value) { this.replaceChildren(String(value ?? "")); }
  get elements() { return this.children.filter((c) => c instanceof FakeElement); }
  get lastElementChild() { return this.elements.at(-1) ?? null; }
  *walk() { for (const c of this.elements) { yield c; yield* c.walk(); } }
  querySelector(selector) {
    if (!/^\.[\w-]+$/.test(selector)) throw new Error(`unsupported selector ${selector}`);
    for (const node of this.walk()) if (node.classList.contains(selector.slice(1))) return node;
    return null;
  }
  findAll(tag) { return [...this.walk()].filter((n) => n.tagName === tag.toUpperCase()); }
  scrollIntoView() {}
  focus() {}
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  dispatch(type) {
    const event = { type, target: this, preventDefault() {}, key: "", ctrlKey: false, metaKey: false };
    for (const fn of this.listeners.get(type) || []) fn(event);
  }
  requestSubmit() { this.dispatch("submit"); }
}

const byId = new Map();
for (const match of html.matchAll(/<(\w+)\b([^>]*)\sid="([^"]+)"([^>]*)>/g)) {
  const [, tag, before, id, after] = match;
  if (tag === "script") continue;
  const attrs = `${before} ${after}`;
  const node = new FakeElement(tag);
  node.setAttribute("id", id);
  node.hidden = /\shidden\b/.test(attrs);
  node.disabled = /\sdisabled\b/.test(attrs);
  byId.set(id, node);
}

const body = new FakeElement("body");
const jaegerMeta = { content: scenario.jaegerMeta ?? (html.match(/name="helpdesk-jaeger-ui-url" content="([^"]*)"/)?.[1] ?? "") };
const document = {
  baseURI: "http://page.test/",
  body,
  getElementById: (id) => byId.get(id) ?? null,
  createElement: (tag) => new FakeElement(tag),
  querySelector(selector) {
    if (selector === 'meta[name="helpdesk-jaeger-ui-url"]') return jaegerMeta;
    throw new Error(`unsupported document selector ${selector}`);
  },
};

const storage = new Map(Object.entries(scenario.storage ?? {}));
const localStorage = {
  getItem: (k) => (storage.has(k) ? storage.get(k) : null),
  setItem: (k, v) => { storage.set(k, String(v)); },
};

// --- scripted network -----------------------------------------------------------------

const log = [];
const routes = new Map(Object.entries(scenario.routes ?? {}).map(([k, v]) => [k, [...v]]));
const encoder = new TextEncoder();
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function streamBody(key, spec) {
  const chunks = [...(spec.chunks ?? [])];
  let pings = spec.pings ?? 0;
  let cancelled = false;
  return new ReadableStream({
    async pull(controller) {
      await wait(spec.delayMs ?? 1);
      if (chunks.length) { controller.enqueue(encoder.encode(chunks.shift())); return; }
      // "linger": the server keeps the connection open with comments after `done`.
      if (pings > 0) {
        pings -= 1;
        await wait(spec.pingMs ?? 200);
        if (!cancelled) controller.enqueue(encoder.encode(": ping\n\n"));
        return;
      }
      if (spec.hold) { await new Promise(() => {}); }
      if (spec.reset) { controller.error(new TypeError("network error")); return; }
      controller.close();
    },
    cancel() { cancelled = true; log.push({ event: "cancel", key }); },
  });
}

async function fakeFetch(input, init = {}) {
  if (input instanceof URL) {
    // Fixture mode reads files next to the page.
    const file = new URL(input.pathname.slice(1), new URL(`file://${indexPath}`));
    try { return new Response(readFileSync(file), { status: 200 }); } catch { return new Response(null, { status: 404 }); }
  }
  const method = (init.method || "GET").toUpperCase();
  const key = `${method} ${input}`;
  log.push({ event: "fetch", key, body: init.body ? JSON.parse(init.body) : null });
  const queue = routes.get(key);
  if (!queue || queue.length === 0) return new Response(null, { status: 404 });
  const spec = queue.length > 1 ? queue.shift() : queue[0];
  if (spec.networkError) throw new TypeError("Failed to fetch");
  const headers = { "content-type": spec.contentType ?? (spec.chunks ? "text/event-stream" : "application/json") };
  const payload = spec.chunks || spec.hold ? streamBody(key, spec) : spec.json === undefined ? null : JSON.stringify(spec.json);
  return new Response(payload, { status: spec.status ?? 200, headers });
}

// --- run --------------------------------------------------------------------------------

const errors = [];
process.on("unhandledRejection", (err) => errors.push(String(err && err.stack || err)));

const context = vm.createContext({
  document, localStorage, fetch: fakeFetch, location: { search: scenario.search ?? "" },
  Response, ReadableStream, TextDecoder, TextEncoder, URL, URLSearchParams,
  setTimeout, clearTimeout, crypto, console,
});

function snapshot(label) {
  const text = (id) => byId.get(id).textContent;
  return {
    label,
    busy: body.getAttribute("aria-busy") === "true",
    status: text("status-line"),
    statusKind: byId.get("status-line").dataset.kind ?? null,
    disabled: Object.fromEntries(["send", "approve", "decline", "new-session", "user-id", "session-id"]
      .map((id) => [id, byId.get(id).disabled])),
    pendingHidden: byId.get("pending").hidden,
    pendingSummary: text("pending-summary"),
    messages: byId.get("messages").elements.map((li) => ({ class: li.getAttribute("class"), text: li.textContent })),
    timeline: text("timeline"),
    trace: text("trace-id"),
    traceLinks: [...byId.get("trace-id").findAll("a"), ...byId.get("messages").findAll("a"), ...byId.get("timeline").findAll("a")]
      .map((a) => a.getAttribute("href")),
    identityError: text("identity-error"),
  };
}

async function idle(timeoutMs) {
  const started = Date.now();
  // Let init() and any queued work start before checking.
  await wait(5);
  while (body.getAttribute("aria-busy") === "true") {
    if (Date.now() - started > timeoutMs) return false;
    await wait(5);
  }
  await wait(5);
  return true;
}

function parseWithPage(chunksB64) {
  const events = [];
  const parser = context.createSseParser((name, data) => events.push([name, data]));
  const decoder = new TextDecoder();
  for (const chunk of chunksB64) parser.push(decoder.decode(Buffer.from(chunk, "base64"), { stream: true }));
  parser.push(decoder.decode());
  parser.end();
  return events;
}

const report = { log, snapshots: [], idle: [], parsed: [], errors };
try {
  vm.runInContext(scriptBlock("sse-parser"), context, { filename: "index.html#sse-parser" });
  if (scenario.parse) {
    report.parsed = scenario.parse.map(parseWithPage);
  } else {
    vm.runInContext(scriptBlock("app"), context, { filename: "index.html#app" });
    for (const step of scenario.steps ?? []) {
      switch (step.op) {
        case "idle": report.idle.push(await idle(step.timeoutMs ?? 3000)); break;
        case "sleep": await wait(step.ms); break;
        case "snapshot": report.snapshots.push(snapshot(step.label)); break;
        case "send": byId.get("message").value = step.text; byId.get("composer").requestSubmit(); break;
        case "click": byId.get(step.id).dispatch("click"); break;
        case "set": byId.get(step.id).value = step.value; byId.get(step.id).dispatch("change"); break;
        // Typing without leaving the field: no "change" event yet.
        case "type": byId.get(step.id).value = step.value; break;
        default: throw new Error(`unknown step ${step.op}`);
      }
    }
  }
} catch (err) {
  errors.push(String(err && err.stack || err));
}
process.stdout.write(JSON.stringify(report));
// A held-open stream would otherwise keep the process alive.
process.exit(0);
