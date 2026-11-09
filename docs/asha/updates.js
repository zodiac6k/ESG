// Reads updates.md and shows each "## date | title" block as a card.
// Asha only edits updates.md, never this file.

const SHOW_FIRST = 5;   // how many updates to show before "Show older updates"
const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Turn "**bold**" into bold text (after escaping, so it is safe)
function inline(s) {
  return escapeHtml(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
}

function renderBody(lines) {
  let html = "", inList = false;
  for (const raw of lines) {
    const line = raw.trim();
    if (line.startsWith("- ")) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += "<li>" + inline(line.slice(2)) + "</li>";
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      if (line) html += "<p>" + inline(line) + "</p>";
    }
  }
  if (inList) html += "</ul>";
  return html;
}

// "2026-11-09" -> { day: "9", mon: "Nov", year: "2026", ago: "today" }
function describeDate(iso) {
  const m = iso.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);
  if (!m) return { day: "•", mon: "", label: iso, ago: "" };
  const d = new Date(+m[1], +m[2] - 1, +m[3]);
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const diff = Math.round((today - d) / 86400000);
  const ago = diff === 0 ? "Today" : diff === 1 ? "Yesterday" : diff > 1 && diff < 7 ? diff + " days ago" : "";
  const label = d.toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long", year: "numeric" });
  return { day: String(+m[3]), mon: MONTHS[+m[2] - 1], label, ago };
}

function card(u, i) {
  const d = describeDate(u.date);
  return `
    <article class="update reveal show${i === 0 ? " latest" : ""}">
      <div class="day"><b>${d.day}</b><small>${d.mon}</small></div>
      <div class="body">
        <div class="meta">${escapeHtml(d.label)}${d.ago ? " · " + d.ago : ""}${i === 0 ? '<span class="badge">Latest</span>' : ""}</div>
        <h3>${inline(u.title)}</h3>
        ${renderBody(u.body)}
      </div>
    </article>`;
}

fetch("updates.md", { cache: "no-store" })
  .then(r => { if (!r.ok) throw new Error(r.status); return r.text(); })
  .then(text => {
    // Remove <!-- ... --> notes so the template is not shown, then split at "## "
    const clean = text.replace(/<!--[\s\S]*?-->/g, "");
    const items = clean.split(/^## /m).slice(1).map(block => {
      const [heading, ...rest] = block.split("\n");
      const [date, ...titleParts] = heading.split("|");
      return { date: date.trim(), title: titleParts.join("|").trim(), body: rest };
    });
    // Newest first (dates written as YYYY-MM-DD sort correctly as text)
    items.sort((a, b) => b.date.localeCompare(a.date));

    const list = document.getElementById("update-list");
    const more = document.getElementById("more-btn");
    if (!items.length) { list.innerHTML = '<p class="muted">No updates yet.</p>'; return; }

    document.getElementById("update-meta").textContent =
      `${items.length} update${items.length > 1 ? "s" : ""} so far · newest first`;

    let shown = 0;
    function showMore() {
      list.insertAdjacentHTML("beforeend", items.slice(shown, shown + SHOW_FIRST).map((u, k) => card(u, shown + k)).join(""));
      shown += SHOW_FIRST;
      more.hidden = shown >= items.length;
    }
    list.innerHTML = "";
    showMore();
    more.addEventListener("click", showMore);
  })
  .catch(() => {
    document.getElementById("update-list").innerHTML =
      '<p class="muted">Updates could not be loaded. (If you opened this file directly from your computer, use the live website link instead.)</p>';
  });
