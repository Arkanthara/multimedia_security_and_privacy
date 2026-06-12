
#show figure.where(kind: "subfigure"): set figure(supplement: "Figure")

#show figure.where(kind: image): outer => {
  counter(figure.where(kind: "subfigure")).update(0)
  set figure(numbering: (..nums) => {
    let outer-nums = counter(figure.where(kind: image)).at(outer.location())
    std.numbering("1a", ..outer-nums, ..nums)
  })
  show figure.where(kind: "subfigure"): inner => {
    show figure.caption: it => context {
      std.numbering("(a)", it.counter.at(inner.location()).last())
      [ ]
      it.body
    }
    inner
  }
  outer
}
// Main report file

#import "@preview/touying:0.7.3": *
#import themes.university: *
#import themes.stargazer: *
#import "@preview/chronos:0.3.0": *
#import "@preview/pintorita:0.1.4"

#import "@preview/theorion:0.6.0": *
#import cosmos.clouds: *
#show: show-theorion

#show raw.where(lang: "pintora"): it => pintorita.render(it.text)

#import "@preview/numbly:0.1.0": numbly

// #context pdfpc.pdfpc-file(here())

#show: stargazer-theme.with(
  aspect-ratio: "16-9",
  // Fix header logo: box it with explicit height and vertical alignment  
  header-right: self => {  
    box(utils.display-current-heading(level: 1))  
    h(.2em)  
    box(height: 1.5em, baseline: 30%, image("img/unige.svg", height: 1.5em))  
  },  
  config-info(
    title: [Watermarking Spatial method],
    subtitle: [],
    author: [Michel Jean Joseph Donnet],
    date: datetime.today(),
    institution: [Faculty of Science, University of Geneva],
    contact: [],
    logo: image("./img/unige.svg", height: 2em),
  ),
    // transparence du décor de fond
  alpha: 12%,

  // barre de progression discrète
  progress-bar: true,

  // Palette personnalisée
  config-colors(
    // Couleur institutionnelle adoucie
    // primary: rgb("#CF0063"),
    primary: rgb("#0A7A66"),

    // Version sombre pour slides focus / contrastes
    // primary-dark: rgb("#7A0042"),
    primary-dark: rgb("#065345"),

    // Couleur du texte sur fonds colorés
    secondary: rgb("#FFFFFF"),

    // Couleur faculté (vert sarcelle)
    tertiary: rgb("#0A7A66"),

    // Fond général légèrement cassé
    neutral-lightest: rgb("#FAFAFA"),

    // Texte principal
    // neutral-darkest: rgb("#7A0042"),
    neutral-darkest: rgb("#032e26"),
  ),
)


// #set heading(numbering: numbly("{1}.", default: "1.1"))

#show raw.where(block: true): set block(fill: luma(240), inset: 1em, radius: 0.5em, width: 100%)
// #show raw.where(block: false): set block(fill: luma(240), inset: 1em, radius: 0.5em, width: 100%)
// #show raw.where(block: false): box.with(
//   fill: rgb("#e573e927"),
//   inset: (x: 3pt, y: 0pt),
//   outset: (y: 3pt),
//   radius: 2pt,
// )

// #import "graph_utils.typ": *
#import "neural-viz/lib.typ": *
#import emoji: camera

#import "@preview/algorithmic:1.0.7"
#import algorithmic: style-algorithm, algorithm-figure

#show: style-algorithm

#show raw.where(block: true): set block(fill: luma(240), inset: 1em, radius: 0.5em, width: 100%)

#let src_dir = "../../.."

#title-slide()

#outline-slide()

= Introduction

#figure(
  {
    let nodes = (
      image-node(0, src: src_dir + "/img/tangled_2.png", cover: true, title: "", image-size: (5, 6), pos: explicit-pos(0, y: 0)),
      image-node(1, src: src_dir + "/img/tangled.png", cover: true, title: "", image-size: (5, 6), pos: explicit-pos(-1, y: 0.5)),
      image-node(2, src: src_dir + "/img/tangled_3.png", cover: true, title: "", image-size: (5, 6), pos: explicit-pos(1.3, y: -0.5)),
      image-node(3, src: src_dir + "/img/tangled_4.png", cover: true, title: "", image-size: (8, 6), pos: explicit-pos(-1, y: -0.6)),
      image-node(4, src: src_dir + "/img/tangled_5.png", cover: true, title: "", image-size: (5, 6), pos: explicit-pos(1.5, y: 0.6)),
      image-node(5, src: src_dir + "/img/tangled_6.png", cover: true, title: "", image-size: (5, 6), pos: explicit-pos(0, y: 1)),
      text-node(6, "?", size: 5em, pos: explicit-pos(-2, y: 0)),
      text-node(7, "?", size: 5em, pos: explicit-pos(4, y: 0)),
      text-node(8, "?", size: 5em, pos: explicit-pos(3, y: 1)),
      text-node(9, "?", size: 5em, pos: explicit-pos(-1.7, y: 1)),
      text-node(10, "My ?", size: 2em, pos: explicit-pos(3, y: -0.6)),
      text-node(11, "Your ?", size: 2em, pos: explicit-pos(-2, y: -0.6)),

    )
    ml-diagram(nodes, spacing: 1em, label-size: 0.7em)
  }
)

#pdfpc.speaker-note(
  "
  Watermark:
  - Data protection
  - Intellectual property
  - Attribution
  - Traceability

  Challenge:
  - Robustness to transformations
  - Imperceptibility
  - Capacity
  - Efficiency
  "
)

= Embedding

#figure(
  {
    let nodes = (
      table-node("init", title: "Initial patch", caption-pos: "top", title-gap: 0.5em, rows: 1, columns: 1, cell-size: (1,1), cells: ([]), cell-aligns: ((center + horizon,)),  cell-shift: (0em, -0.2em), cell-text-size: 1em),
      table-node("p", title: "", rows: 1, columns: 1, cell-size: (1,1), cells: ([p]), cell-aligns: ((center + horizon,)),  cell-shift: (0em, -0.2em), cell-text-size: 1em, pos: below("init", by: 2)),
      table-node("p_upsampled", title: "", rows: 1, columns: 1,  cell-size: (2,2), cells: ([p]), cell-aligns: ((center + horizon,)), cell-shift: (0em, -0.4em), cell-text-size: 2em, pos: below("p")),
      table-node(
        "p_padded",
        title: "",
        rows: 2,
        columns: 2,
        cell-size: (2, 2),
        cells: ([p], [q], [b], [d]),
        cell-aligns: (center + horizon, center + horizon, center + horizon, center + horizon),
        cell-shift: (0em, -0.5em), // base shift for all cells
        row-shifts: ((0em, 0em), (0em, 0.5em)), // second row only
        cell-text-size: 2em,
        pos: below("p_upsampled")
      ),
      table-node(
        "p_final",
        title: "",
        rows: 4,
        columns: 3,
        cell-size: (2, 2),
        cells: ([p], [q], [p], [b], [d], [b], [p], [q], [p], [b], [d], [b]),
        cell-aligns: (center + horizon, center + horizon, center + horizon, center + horizon),
        cell-shift: (0em, -0.5em), // base shift for all cells
        row-shifts: ((0em, 0em), (0em, 0.5em), (0em, 0em), (0em, 0.5em)), // second row only
        cell-text-size: 2em,
        pos: right-of("p_padded", by: 5),
      ),


      image-node("tangled", src: src_dir + "/img/tangled_2.png", cover: true, title: "", image-size: (6, 8), pos: right-of("p_final", by: 4)),
      table-node(
        "p_final_embedded",
        title: "",
        rows: 4,
        columns: 3,
        cell-size: (2, 2),
        cells: ([p], [q], [p], [b], [d], [b], [p], [q], [p], [b], [d], [b]),
        cell-aligns: (center + horizon, center + horizon, center + horizon, center + horizon),
        cell-shift: (0em, -0.5em), // base shift for all cells
        row-shifts: ((0em, 0em), (0em, 0.5em), (0em, 0em), (0em, 0.5em)), // second row only
        cell-text-size: 2em,
        pos: right-of("p_final", by: 4),
      ),
    )
    let edges = (
      ml-edge("init", "p", label: "Watermark embedding"),
      ml-edge("p", "p_upsampled", label: "Upsampling"),
      ml-edge("p_upsampled", "p_padded", label: "Padding"),
      ml-edge("p_padded", "p_final", label: "Image padding"),
      ml-edge("p_final", "p_final_embedded", label: "Embedding"),

    )
    ml-diagram(nodes, edges: edges, spacing: 1em, label-size: 0.7em)
  }
)

= Decoding

#tblock(title: "Decoding process")[
  - Denoising
  - Direct decoding (baseline)
  - Synchronization
  - Patch extraction
  - Message recovery
  - Return best of both decoding paths
]

#pagebreak()

#tblock(title: "Synchronization process")[
  - Crop (for speed)
  - Autocorrelation
  - Peak detection
  - Select reference peaks
  - Get rotation and scaling parameters
  - Apply inverse transform
  - Get translation and patch orientation
  - Apply inverse transform
]

#pagebreak()

#tblock(title: "Patch extraction process")[
  - Divide residual into patches
  - Sum up patches
  - positive = 1, negative = 0
  - Get confidence score (mean of absolute values in patch)
  - Return bit sequence and confidence scores
]

// #figure(
//   {
//     let nodes = (
//       image-node("tangled", src: src_dir + "/img/tangled_2.png", cover: true, title: "", image-size: (3, 4)),
//       arrow-node("denoising", title: "Denoising", pos: right-of("tangled")),
//       arrow-node("autocor", title: "Autocorrelation", pos: right-of("denoising")),
//       arrow-node("peak-detection", title: "Peak detection", pos: right-of("autocor")),
//       arrow-node("synchronization", title: "Synchronization", pos: right-of("peak-detection")),
//       module("lsb-extraction", title: "LSB extraction", pos: right-of("synchronization")),
//       module("bit-decoding", title: "Bit decoding", pos: right-of("lsb-extraction")),
//     )
//     ml-diagram(nodes, spacing: 1em, label-size: 0.7em)
//   }
// )



// #bibliography("bibliography.bib")