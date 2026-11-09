# How to Build Asha's Yoga Website: Step by Step (for an 8th Grader)

You need **no software** for this. Only a web browser and a free GitHub account.

---

## Part A: Understand the idea (5 minutes)

A website is a set of **text files** that a browser reads.

| File | What it does | Real-life comparison |
|---|---|---|
| `index.html` | The **content**: headings, paragraphs, buttons | The bricks and rooms of a house |
| `style.css` | The **look**: colours, fonts, spacing | The paint and furniture |
| `updates.js` | The **behaviour**: reads the daily notes and shows them | The electricity |
| `updates.md` | The **daily diary**: the only file Asha edits each day | A notebook on the wall |

**GitHub** stores the files. **GitHub Pages** turns them into a live website for free.

---

## Part B: Build the page skeleton with HTML

### Step 1: Every HTML page starts the same way
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Asha Ramaswamy Yoga</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  ... everything you see goes here ...
</body>
</html>
```
- `<head>` = information **about** the page (title, which CSS to use).
- `<body>` = what the visitor **sees**.
- The `viewport` line makes it work on phones.

### Step 2: Learn 6 tags. That is enough.
| Tag | Meaning | Example |
|---|---|---|
| `<h1>` `<h2>` | Big and medium headings | `<h1>Yoga for Well-Being</h1>` |
| `<p>` | Paragraph | `<p>45-minute sessions.</p>` |
| `<a href="">` | Link | `<a href="tel:+919342113730">Call</a>` |
| `<ul><li>` | Bullet list | `<ul><li>Neck rotation</li></ul>` |
| `<table><tr><th><td>` | Table (row, header cell, cell) | Program details table |
| `<section id="">` | A block of the page | `<section id="about">` |

Every tag that opens must close: `<p>` … `</p>`.

### Step 3: Turn Asha's document into sections
Take the two proposal documents (45-min office module and 60-min workshop) and give each heading its own `<section>`:

| Document heading | Website section | `id` |
|---|---|---|
| Title + Executive Summary | Hero (first screen) | `home` |
| Instructor Profile | About Asha | `about` |
| Program Details + Benefits | Program at a Glance | `program` |
| Session Flow + Safety | 45-Minute Office Module | `session` |
| 2nd document: 60-min Workshop Agenda + Postures + Organiser Guidelines | 60-Minute Workshop | `workshop` |
| *(new)* | Daily Updates | `updates` |
| *(new)* | Contact | `contact` |

### Step 4: Add the menu
Links that start with `#` jump to a section on the same page:
```html
<nav>
  <a href="#about">About</a>
  <a href="#updates">Daily Updates</a>
  <a href="#contact">Contact</a>
</nav>
```

### Step 5: Add contact links that *do* something
```html
<a href="tel:+919342113730">+91 93421 13730</a>           <!-- opens the phone dialer -->
<a href="https://wa.me/919342113730">WhatsApp</a>          <!-- opens WhatsApp chat -->
<a href="mailto:kscasha@gmail.com">kscasha@gmail.com</a>   <!-- opens email app -->
```
Note: for `wa.me`, write the number with **no +, no spaces**.

---

## Part C: Make it pretty with CSS

### Step 6: The CSS rule pattern
```css
selector { property: value; }
```
Example: make every `h2` heading orange-brown:
```css
h2 { color: #8f3f1b; }
```

### Step 7: Store colours in one place (variables)
```css
:root {
  --accent: #b5562b;
}
.btn { background: var(--accent); }
```
Change `--accent` once and every button changes colour.

### Step 8: Cards that sit side by side and stack on phones
```css
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 16px;
}
```
In plain English: "fit as many 240px-wide boxes as you can in a row. If the screen is narrow, put them one under another."

### Step 9: Phone check
```css
@media (max-width: 600px) {
  section { padding: 40px 0; }
}
```
This means: "Only on screens narrower than 600px, use smaller spacing."

---

## Part D: Daily updates with a little JavaScript

### Step 10: The idea
Asha should never edit HTML. She writes in a simple diary file `updates.md`:
```
## 2026-11-10 | Neck and shoulder focus
Great energy today, 18 participants.
- Neck movements, 5 rounds each
- Bhramari pranayama for calm
```
Rules:
- `## ` starts a new update
- date in **YEAR-MONTH-DAY** format, then ` | `, then the title
- normal lines become paragraphs, lines starting with `- ` become bullets

### Step 11: What `updates.js` does (in 4 steps)
1. **fetch** (download) `updates.md`
2. **split** it wherever a line starts with `## `
3. **sort** by date, newest first
4. **draw** each one as a card inside `<div id="update-list">`

You can read the code in `updates.js`. Each step has a comment.

---

## Part E: Put it on the internet (GitHub Pages)

### Step 12: Upload the files
1. Sign in to github.com and open the repository.
2. Go to the `docs` folder and open `asha`.
3. To create a file, click **Add file → Create new file**. To add files from your computer, click **Upload files**.
4. Click **Commit changes**.

### Step 13: Switch on GitHub Pages (one time only)
1. Go to the repository's **Settings → Pages**.
2. Under "Source", choose **Deploy from a branch**.
3. Branch: **main**, folder: **/docs**. Click **Save**.
4. After 1–2 minutes the site is live at:
   `https://<your-github-username>.github.io/<repository-name>/asha/`

---

## Part F: Posting the daily update (Asha's 2-minute routine)

### Step 14: Every day after class
1. Open `docs/asha/updates.md` on github.com (on a phone or laptop).
2. Click the **pencil ✏️** (Edit).
3. Paste a new block **at the top of the list**:
   ```
   ## 2026-11-10 | Title of today's class
   One or two sentences about the class.
   - Highlight 1
   - Highlight 2
   ```
4. Click **Commit changes** (green button).
5. Wait 1–2 minutes and refresh the website. The new update shows first.

**Tip:** Save the GitHub mobile app link to your phone's home screen to make this quicker.

---

## Part G: Checklist before you call it "done"

- [ ] Opens on a phone without needing to scroll sideways
- [ ] Phone link opens the dialer; WhatsApp and email links work
- [ ] Today's update is at the top
- [ ] No spelling mistakes in Asha's qualifications
- [ ] Dates use the YYYY-MM-DD format

## Ideas for later (level-up projects)
1. Add photos: upload `photo.jpg` to the folder and add `<img src="photo.jpg" alt="Asha teaching">`.
2. Add a Google Form for bookings and link to it with a button.
3. Buy a custom domain (for example `ashayoga.in`) and connect it in Settings → Pages.
