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
#import "@preview/lilaq:0.6.0" as lq

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
    title: [Multimedia Security and Privacy],
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

= Usages of Digital Watermarking

== Watermarking

#tblock(title: "Definition")[
Ensures ownership and authenticity of digital content
]

#tblock(title: "Requirements")[
- Robustness
- Percebility
- Embedding Capacity ($~$64 - 128 bits)
]

#tblock(title: "Applications")[
  - Copyright protection
  - Real vs synthetic media detection
]



== Steganography

#tblock(title: "Definition")[
Hides the existence of a message for secret communication
]

#tblock(title: "Requirements")[
- Undetectability
- Embedding Capacity ($~$1 Ko per image)
]

#tblock(title: "Applications")[
  - Alternative to cryptography
]


== Tamper-proofing

#tblock(title: "Definition")[
Detects unauthorized modifications to digital content (Key based or compression-based)
]

#tblock(title: "Requirements")[
- Capacity
- Robustness
- Localization
]

#tblock(title: "Applications")[
  - Integrity verification
  - Forensics
]

= Watermarking vs Data Hiding

#tblock(title: "Technologies")[
  - Zero-bit watermarking (Detection problem)
  - Multi-bit watermarking (Decoding problem)
]

== Block diagram

#figure(
  {
  let nodes = (
    dataset(0, title: $t_0$, size: (2, 3)),
    module(1, title: $T$, pos: right-of("0"), size: (2, 2)),
    vector(2, title: $X_0$, pos: right-of("1"), dir: "v", stack: 3, size: (1.5, 3)),
    gate-node(3, pos: right-of("2")),
    vector(6, title: $X_t$, pos: right-of("3"), dir: "v", stack: 3, size: (1.5, 3)),
    module(7, title: $p_theta (hat(X)_0_t, hat(z)_t | X_t, t, Y_epsilon)$, pos: right-of("6"), size: (9, 2)),
    vector(16, title: $hat(X)_0_t$, pos: right-of("7"), dir: "v", stack: 3, size: (1.5, 3)),
    module(11, title: $T^(-1)$, pos: right-of("16"), size: (2, 2)),
    dataset(13, title: $hat(t)_0_t$, pos: right-of("11"), size: (2, 3)),

    compare-node(12, title: $cal(L)_z$, size: (2, 2.5), pos: below("11", by: 1.15)),

    text-node(4, $bold(z)_B$, pos: below("0", by: 1.3)),
    module(5, title: $cal(M)_t$, pos: right-of("4"), size: (2, 2)),
    vector(14, title: $z_t$, pos: right-of("5"), dir: "v", stack: 3, size: (1.5, 3)),

    compare-node(15, title: $cal(L)_x$, size: (2, 2.5), pos: above("11", by: 1.15)),

    dataset(8, title: $Y$, pos: below("4", by: 1.05), size: (2, 3)),
    module(9, title: $epsilon_y$, pos: right-of("8"), size: (2, 2)),
    vector(10, title: $Y_epsilon$, pos: right-of("9"), dir: "v", stack: 3, size: (1.5, 3)),

  )

  let edges = (
    ml-edge("0", "1"),
    ml-edge("1", "2"),
    ml-edge("2", "3"),
    ml-edge("3", "6"),

    ml-edge("4", "5"),
    ml-edge("5", "14"),
    ml-edge("14", "3", orthogonal: true),


    ml-edge("6", "7"),
    ml-edge("8", "9"),
    ml-edge("9", "10"),


    ml-edge("7", "11"),
    ml-edge("11", "13"),
    ml-edge("11", "13"),

    ml-edge("7", "12", orthogonal: "v", from-shift: -0.3, to-shift: -0.15, label: $hat(z)_t$, label-pos: 33%, label-side: left),
    ml-edge("14", "12", to-shift: -0.15),

    ml-edge("10", "7", orthogonal: true, to-shift: -0.3, crossing: true),

    ml-edge("2", "15", to-shift: -0.15, orthogonal: "v"),

    ml-edge("16", "15", orthogonal: "v", to-shift: 0.15),

  )

  ml-diagram(nodes, edges: edges, label-size: 0.7em, spacing: 1.3em)
  },
)

== Block diagram



#figure(
  {
  let size = (4, 5)
  let nodes = (
    image-node("input", src: src_dir + "/img/tangled_2.png", cover: true, title: "", image-size: size, pos: explicit-pos(0, y: 0)),

    module("embedder", title: [Embedder], pos: right-of("input", by: 2)),
    module("encoder", title: [Encoder], pos: above("embedder")),
    group("embedding-encoder", ("embedder", "encoder"), title: ""),

    text-node("m", [Message *$m$*], pos: above("encoder")),
    text-node("key_1", [#emoji.key], pos: below("embedder")),

    image-node("embed", src: src_dir + "/img/tangled_2.png", cover: true, title: "", image-size: size, pos: right-of("embedder")),
    module("public", title: [Public \ multimedia \ channels], pos: right-of("embed")),
    image-node("distorted", src: src_dir + "/img/attacked.png", cover: true, title: "", image-size: size, pos: right-of("public")),

    module("extractor", title: [Extractor], pos: right-of("distorted")),
    module("decoder", title: [Decoder], pos: above("extractor")),
    group("extractor-decoder", ("extractor", "decoder"), title: ""),

    text-node("hat-m", [Message *$hat(m)$*], pos: above("decoder")),
    text-node("key_2", [#emoji.key], pos: below("extractor")),

    text-node("optional", [*$x$*], pos: right-of("extractor")),

  )

  let edges = (
    ml-edge("input", "embedder", label: [*$x$*], label-side: left, label-pos: 25%),
    ml-edge("input", "encoder", mark: "-|>", from-side: "right", to-side: "left", via: ((1, 0), (1, -1))),
    ml-edge("embedder", "embed", label: [*$y$*]),
    ml-edge("embed", "public"),
    ml-edge("public", "distorted"),
    ml-edge("distorted", "extractor", label: [*$hat(y)$*]),

    ml-edge("m", "encoder"),
    ml-edge("encoder", "embedder"),

    ml-edge("extractor", "decoder"),
    ml-edge("decoder", "hat-m"),

    ml-edge("key_1", "embedder", label: [*$k$*]),
    ml-edge("key_2", "extractor", label: [*$k$*]),

    ml-edge("optional", "extractor", mark: "--|>"),


  )

  ml-diagram(nodes, edges: edges, label-size: 1em, spacing: 1.6em)
  },
) 

= Watermark detection

#tblock(title: "Hypothesis testing")[
  - $H_0: y = x$
  - $H_1: y = x + w$
]

#tblock(title: "Score definition")[
  - $S = (y^T w)/M$ ($M$: number of pixels)
  - $S ~ cal(N)(mu_(H_0), sigma_(H_0)^2)$ under $H_0$ (Central Limit Theorem)
  - $S ~ cal(N)(mu_(H_1), sigma_(H_1)^2)$ under $H_1$
]

#tblock(title: "Probabilities")[
  - Correct Detection: $P_D(gamma) = Pr(S >= gamma | H_1)$
  - False Acceptance: $P_(F A)(gamma) = Pr(S >= gamma | H_0)$
]

== Graph hypothesis testing

#let mu1 = 0
#let sigma1 = 1
#let mu2 = 3
#let sigma2 = 1
#let threshold = 1.2

#let xs = lq.linspace(-5, 8, num: 200)
#let xfa = lq.linspace(threshold, 8, num: 200)
#let xmiss = lq.linspace(-5, threshold, num: 200)

#lq.diagram(
  grid: none,
  width: 25cm,
  height: 10cm,
  bounds: "strict",
  lq.plot(
    label: [$H_0$],
    xs,
    xs.map(x =>
      calc.exp(-(x - mu1)*(x - mu1) / (2 * sigma1*sigma1))
    ),
    mark: none,
    stroke: 0.2em
  ),
    lq.plot(
    label: [$H_1$],
    xs,
    xs.map(x =>
      calc.exp(-(x - mu2)*(x - mu2) / (2 * sigma2*sigma2))
    ),
    mark: none,
    stroke: 0.2em
  ),

  lq.fill-between(
    label: [$P_(F A)$],
    xfa,
    xfa.map(x =>
      calc.exp(-(x - mu1)*(x - mu1) / (2 * sigma1*sigma1))
    ),
    fill: rgb(100%, 5%, 5%, 50%),
  ),

  lq.fill-between(
    label: [$P_(text("miss"))$],
    xmiss,
    xmiss.map(x =>
      calc.exp(-(x - mu2)*(x - mu2) / (2 * sigma2*sigma2))
    ),
    fill: rgb("#0dffcb80"),
  ),

  lq.vlines(threshold, stroke: (paint: green, thickness: 0.1em), label: [$gamma$])
)

#tblock(title: "Neyman Pearson Strategy")[
  - Fix $P_(F A)$: $ P_(F A)(gamma) = integral_(> gamma) p(x | H_0) d x $
  - Minimize probability of miss: $ P_(text("miss")) = 1 - P_D = integral_(<= gamma) p(x | H_1) d x $
]

#tblock(title: "Bayesian Strategy")[
  - Fix priors $P(H_0)$ and $P(H_1)$
  - Minimize overall error: $P_(text("error")) = P_(F A) P(H_0) + P_(text("miss")) P(H_1)$
]

