# Multimedia Security and Privacy — Course Website

A bilingual (🇫🇷 French / 🇬🇧 English) static website for the *Multimedia Security and Privacy* course taught by Prof. S. Voloshynovskiy at the University of Geneva.

The site features **light** (Catppuccin Latte) and **dark** (Catppuccin Macchiato) themes, mathematical formulas rendered via MathJax, and CeTZ diagrams compiled via Typst.

---

## 📁 Repository Structure

```
multimedia_security_and_privacy/
├── course/                    # Original lecture PDF slides
│   ├── MMSEC Lecture 1_compressed.pdf
│   ├── MMSEC Lecture 2_compressed.pdf
│   ├── MMSEC Lecture 3 - 2025-05-07_compressed.pdf
│   ├── MMSEC Lecture 3 - 2025-05-15_compressed.pdf
│   └── MMSEC Lecture 4_compressed.pdf
│
├── website/
│   ├── latte.scss             # Catppuccin Latte (light) theme
│   ├── macchiato.scss         # Catppuccin Macchiato (dark) theme
│   │
│   ├── en/                    # English website
│   │   ├── _quarto.yml        # Quarto project config (English)
│   │   ├── _extensions/       # Quarto extensions (diagram filter)
│   │   ├── index.qmd          # Homepage
│   │   ├── ch1.qmd            # Chapter 1 overview
│   │   ├── ch1c.qmd           # Chapter 1 detailed content
│   │   ├── ch1r.qmd           # Chapter 1 synthesis (quick revision)
│   │   ├── ch2.qmd / ch2c.qmd / ch2r.qmd
│   │   ├── ch3.qmd / ch3c.qmd / ch3r.qmd
│   │   └── ch4.qmd / ch4c.qmd / ch4r.qmd
│   │
│   └── fr/                    # French website (same structure)
│       ├── _quarto.yml
│       ├── _extensions/
│       ├── index.qmd
│       └── ch*.qmd
│
└── .github/workflows/
    └── website.yml            # GitHub Actions — build & deploy to GitHub Pages
```

---

## 🔧 Prerequisites

Install the following tools:

| Tool | Version | Purpose |
|------|---------|---------|
| [Quarto](https://quarto.org/docs/get-started/) | ≥ 1.4 | Website builder |
| [Typst](https://github.com/typst/typst/releases) | ≥ 0.12 | CeTZ diagram rendering |
| [TinyTeX](https://yihui.org/tinytex/) | latest | PDF math rendering (optional) |

### Install Quarto

```bash
# macOS (Homebrew)
brew install --cask quarto

# Linux (Debian/Ubuntu)
wget https://github.com/quarto-dev/quarto-cli/releases/latest/download/quarto-linux-amd64.deb
sudo dpkg -i quarto-linux-amd64.deb

# Windows: download installer from https://quarto.org/docs/get-started/
```

### Install Typst

```bash
# macOS (Homebrew)
brew install typst

# Linux / Windows: download from https://github.com/typst/typst/releases
```

---

## 🏗️ Building the Website Locally

### Build the French version

```bash
cd website/fr
quarto render
```

Output is placed in `website/_site/fr/`.

### Build the English version

```bash
cd website/en
quarto render
```

Output is placed in `website/_site/` (root).

### Build both versions

```bash
cd website/fr && quarto render && cd ../en && quarto render
```

### Preview with live reload

```bash
# French
cd website/fr && quarto preview

# English
cd website/en && quarto preview
```

The preview opens automatically at `http://localhost:4848` (or another available port).

---

## 🌐 Deploying to GitHub Pages

The site is automatically built and deployed on every push to the `course` branch via the GitHub Actions workflow (`.github/workflows/website.yml`).

You can also trigger the workflow manually from the **Actions** tab in GitHub → *Quarto Publish* → **Run workflow**.

The deployed site will be available at:

```
https://<username>.github.io/multimedia_security_and_privacy/
```

---

## 📖 Chapter Structure

Each chapter follows a three-page structure:

| Page | File | Description |
|------|------|-------------|
| **Overview** | `chN.qmd` | Quick summary + synthesis callout at the top for fast revision |
| **Detailed content** | `chNc.qmd` | Full course content with CeTZ diagrams and formulas |
| **Synthesis** | `chNr.qmd` | Condensed key formulas and concepts for exam revision |

The **synthesis** is intentionally placed at the *top* of each overview page (`chN.qmd`) so you don't need to scroll to find it.

---

## 🎨 Themes

| Theme | Mode | Base color |
|-------|------|-----------|
| [Catppuccin Latte](https://github.com/catppuccin/catppuccin) | ☀️ Light | `#eff1f5` |
| [Catppuccin Macchiato](https://github.com/catppuccin/catppuccin) | 🌙 Dark | `#24273a` |

Switch between themes using the toggle in the top-right corner of the website.

---

## 🔍 Adding New Content

1. Create a new `.qmd` file in `website/fr/` or `website/en/`
2. Add it to the sidebar in the corresponding `_quarto.yml`
3. For CeTZ diagrams, use the `{.cetz}` code block type (handled by the `pandoc-ext/diagram` filter)

Example CeTZ diagram:

````markdown
```{.cetz}
#cetz.canvas({
  import cetz.draw: *
  rect((0,0), (3,2), fill: blue.lighten(60%), stroke: blue)
  content((1.5, 1), [My Box])
})
```
````

---

## 📐 Math Formulas

Inline math: `$formula$`

Display math:
```markdown
$$
\text{PSNR} = 10 \cdot \log_{10}\left(\frac{255^2}{\text{MSE}}\right)
$$
```

---

## 📜 License

See [LICENSE](LICENSE).
