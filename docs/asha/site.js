// Small helpers for the whole page. Asha never needs to edit this file.

// 1. Footer year
document.getElementById("year").textContent = new Date().getFullYear();

// 2. Phone menu: the ☰ button opens and closes the menu
const menuBtn = document.querySelector(".menu-btn");
const menu = document.getElementById("menu");
menuBtn.addEventListener("click", () => {
  const open = menu.classList.toggle("open");
  menuBtn.setAttribute("aria-expanded", open);
  menuBtn.textContent = open ? "✕" : "☰";
});
menu.querySelectorAll("a").forEach(a => a.addEventListener("click", () => {
  menu.classList.remove("open");
  menuBtn.textContent = "☰";
  menuBtn.setAttribute("aria-expanded", false);
}));

// 3. Gentle fade-in when a block scrolls into view
const targets = document.querySelectorAll("section .wrap > *:not(.hero-grid)");
if ("IntersectionObserver" in window) {
  const io = new IntersectionObserver(entries => {
    entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add("show"); io.unobserve(e.target); } });
  }, { threshold: 0.08 });
  targets.forEach(t => { t.classList.add("reveal"); io.observe(t); });
}
