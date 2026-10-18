// Shared helpers for every page: talking to the server, naming items, building elements.

// Call the backend. Throws an Error with the server's own message when something fails.
async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* not JSON */ }
    const err = new Error(message);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

// "macos_43_blue_striped_polo.png" -> "Blue striped polo"
function itemName(item) {
  let name = (item.original_filename || "item").replace(/\.[a-z0-9]+$/i, "");
  name = name.replace(/^macos_\d+_/, "").replace(/^[0-9a-f-]{36}_/, "");
  if (!/\s/.test(name)) name = name.replace(/[_-]+/g, " ");   // file names; names you typed stay as typed
  name = name.trim();
  name = name.replace(/\baf1\b/i, "AF1");
  return name.charAt(0).toUpperCase() + name.slice(1);
}

// image_path is relative to the project folder; the server serves those folders as-is.
function imageUrl(item) {
  return "/" + String(item.image_path || "").split("/").map(encodeURIComponent).join("/");
}

// Small version (400px) for grids and thumbnails; the server makes and caches it.
// The big outfit board and the item page keep the full photo.
function thumbUrl(item) {
  const path = String(item.image_path || "");
  return /^(macos_generate|uploads)\//.test(path)
    ? "/thumb/" + path.split("/").map(encodeURIComponent).join("/") : imageUrl(item);
}

// Sort by the number in "macos_12_..." so 2 comes before 10; other names alphabetically.
function byItemNumber(a, b) {
  const num = (it) => { const m = /^macos_(\d+)_/.exec(it.original_filename || ""); return m ? +m[1] : Infinity; };
  return num(a) - num(b) || itemName(a).localeCompare(itemName(b));
}

// Colour-coding class: top / bottom / footwear
function categoryClass(item) {
  return ["top", "bottom", "footwear"].includes(item.category) ? item.category : "";
}

// Days since an item was last worn (null = never worn since tracking began).
function daysSinceWorn(item) {
  if (!item.last_worn) return null;
  const then = new Date(item.last_worn + "T00:00:00");
  return Math.floor((Date.now() - then.getTime()) / 86400000);
}
const UNWORN_DAYS = 30;
// Worn before, but not in the last 30 days (never-worn items don't count: tracking is new)
const isUnworn = (item) => { const d = daysSinceWorn(item); return d !== null && d >= UNWORN_DAYS; };
function lastWornText(item) {
  const d = daysSinceWorn(item);
  return d === null ? "Not worn yet" : d === 0 ? "Worn today" : d === 1 ? "Worn yesterday" : `Last worn ${d} days ago`;
}

// Short readable tag line, e.g. "Polo, blue, striped"
function tagLine(item) {
  const SINGULAR = { "t-shirts": "T-shirt", polos: "Polo", shirts: "Shirt", hoodies: "Hoodie",
    sweatshirts: "Sweatshirt", jackets: "Jacket" };  // jeans, shorts, shoes... stay plural
  const sub = SINGULAR[item.subcategory] || item.subcategory || "";
  const colour = item.color === "multicolor" && item.colors && item.colors.length
    ? item.colors.join(" / ") : item.color;
  const parts = [sub, colour];
  if (item.pattern && !["solid", "none"].includes(item.pattern)) parts.push(item.pattern);
  return parts.filter(Boolean).join(", ").replace(/^./, (c) => c.toUpperCase());
}

// Tiny element builder: el("a", {href: "/", class: "x"}, "text", childElement)
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "style") node.style.cssText = value;
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

// Show a message box inside a container (kind: "" or "error").
function showMessage(container, text, kind = "") {
  container.replaceChildren(el("p", { class: `message ${kind}`, role: kind === "error" ? "alert" : "status" }, text));
}

const ICONS = {
  up: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 10v11H4V10zM7 10l4-7a2 2 0 0 1 3 2l-1 5h6a2 2 0 0 1 2 2.3l-1.4 7A2 2 0 0 1 17.6 21H7"/></svg>',
  down: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M17 14V3h3v11zM17 14l-4 7a2 2 0 0 1-3-2l1-5H5a2 2 0 0 1-2-2.3l1.4-7A2 2 0 0 1 6.4 3H17"/></svg>',
};
function icon(name) {
  const span = document.createElement("span");
  span.innerHTML = ICONS[name]; // fixed SVG strings above, never user data
  return span.firstChild;
}
