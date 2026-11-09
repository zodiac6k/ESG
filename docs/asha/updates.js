// Reads updates.md and shows each "## date | title" block as a card.
// Asha only edits updates.md, never this file.

document.getElementById("year").textContent = new Date().getFullYear();

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderBody(lines) {
  let html = "", inList = false;
  for (const raw of lines) {
    const line = raw.trim();
    if (line.startsWith("- ")) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += "<li>" + escapeHtml(line.slice(2)) + "</li>";
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      if (line) html += "<p>" + escapeHtml(line) + "</p>";
    }
  }
  if (inList) html += "</ul>";
  return html;
}

fetch("updates.md", { cache: "no-store" })
  .then(r => { if (!r.ok) throw new Error(r.status); return r.text(); })
  .then(text => {
    // Split the file at every line that starts with "## "
    // (first remove <!-- ... --> notes so the template is not shown)
    const clean = text.replace(/<!--[\s\S]*?-->/g, "");
    const blocks = clean.split(/^## /m).slice(1);
    const items = blocks.map(block => {
      const [heading, ...rest] = block.split("\n");
      const [date, ...titleParts] = heading.split("|");
      return { date: date.trim(), title: titleParts.join("|").trim(), body: rest };
    });
    // Newest first (dates written as YYYY-MM-DD sort correctly as text)
    items.sort((a, b) => b.date.localeCompare(a.date));

    const list = document.getElementById("update-list");
    if (!items.length) { list.innerHTML = '<p class="muted">No updates yet.</p>'; return; }
    list.innerHTML = items.map(u => `
      <article class="update">
        <div class="date">${escapeHtml(u.date)}</div>
        <h3>${escapeHtml(u.title)}</h3>
        ${renderBody(u.body)}
      </article>`).join("");
  })
  .catch(() => {
    document.getElementById("update-list").innerHTML =
      '<p class="muted">Updates could not be loaded. (If you opened this file directly from your computer, use the live website link instead.)</p>';
  });
