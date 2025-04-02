// 公式便捷输入包
#import "@preview/physica:0.9.3": *
// 带颜色定理环境包
#import "@preview/ctheorems:1.1.3": *
#show: thmrules

// 编号重新定义包
#import "@preview/i-figured:0.2.3"

// 绘图包
#import "@preview/cetz:0.3.1"
#import "@preview/cetz-venn:0.1.1"

// 自定义数学简写
#import "@preview/quick-maths:0.1.0": shorthands

//#import "@preview/indenta:0.0.3": fix-indent

#let ee = {math.upright[e]}
#let ii = {math.upright[i]}

#let cases(..args) = math.cases(..args.pos().map(math.display),gap: 1em);

#let scr(it) = text(
  features: ("ss01",),
  box($cal(it)$)
)

#let alert(it) = underline(highlight(it))

#let eqBlock(body, fill: rgb("#e8f8e8")) = {
 block(
  width: 100%,
  inset: 0.5em,
  radius: 0.3em,
  breakable: false,
  fill: fill,
 )[#body]
}

#let codeBlock(body) = {
 block(
  width: 100%,
  inset: 1.2em,
  radius: 0.3em,
  breakable: true,
  fill: luma(240)
  )[#body]
}

// 定义定理环境
#let theorem = thmbox(
  "theorem",
  "Theorem",
  base_level: 2,
  fill: rgb("#e8e8f8")
)
#let lemma = thmbox(
  "theorem", // Lemmas 和 Theorem 使用相同的计数器 
  "Lemma",
  base_level: 2,
  fill: rgb("#efe6ff")
)
#let corollary = thmbox(
  "corollary",
  "Corollary",
  base_level: 2,
  base: "theorem",      // Corollaries are 'attached' to Theorems
  fill: rgb("#f8e8e8")
)

#let definition = thmbox(
  "definition",
  "Definition",
  base_level: 2,
  breakable: true,
  fill: rgb("#e8f8e8")
)

#let note = thmbox(
  "note",
  "Note",
  stroke: rgb("#aaaaaa") + 1pt,
  base_level: 1,
).with(numbering: none)

#let mark = thmbox(
  "mark",
  "Mark",
  stroke: rgb("#aaaaaa") + 1pt,
  base_level: 1,
).with(numbering: none)

#let exercise = thmbox(
  "exercise",
  "Exercise",
  stroke: rgb("#ffaaaa") + 1pt,
  base_level: 1,
  breakable: true,
).with(numbering: "1.I")

#let example = thmplain("example", "Example").with(numbering: none)

#let remark = thmplain(
  "remark",
  "Remark",
  inset: 0em,
  bodyfmt: x => emph(x),
).with(numbering: none)

#let proof = thmproof(
  "proof",
  "Proof",
  base: "theorem",
)

#let solution = thmplain(
  "solution",
  "Solution",
  base: "exercise",
  inset: 0em,
).with(numbering: none)


// ctex 文本大小定义
#let tiny = 5.5pt
#let scriptsize = 6.5pt
#let footnotesize = 7.5pt
#let small = 9pt
#let normalsize = 10.5pt
#let large = 12pt
#let Large = 15pt
#let LARGE = 18pt
#let huge = 22pt
#let Huge = 26pt

// 创建两个状态，可以在文档中更新
#let page_number_display = state("page_number_display","1") // 页码显示方式



// 另一个 fakepar 实现
#let fakepar=context{box();v(-measure(block()+block()).height)}

// 每一个文件都需要设置的内容
#let module(doc) = [
  // 在此设置防止 LSP 误报
  #set heading(numbering: "1.1  ")
  // 数学符号修改
  #show math.equation: set text(features: ("cv01",))
  // 引用设置
  // 设置引用名称
  #set heading(supplement: [§] + h(-0.25em))
  #set math.equation(supplement: [式])
  // 设置编号使得自动添加 Section 编号
  #show heading: i-figured.reset-counters
  #show math.equation: i-figured.show-equation.with(
    level: 2,
    zero-fill: true,
    leading-zero: true,
    numbering: "(1-1)",
    prefix: "eq:",
    only-labeled: true,  // 只编号有 label 的公式
    unnumbered-label: "", // 不编号的 label
  )
  #show figure.where(kind: image).or(figure.where(kind: table)): i-figured.show-figure.with(
    level: 1,
    zero-fill: true,
    leading-zero: true,
    numbering: "1.1-1",
  )

  // 在任意段落前加上假段落
  // 在 Bug 修复后应该可以去掉
  #show heading: it => it + fakepar
  #show figure: it => it + fakepar
  // #show list: it => it + fakepar
  // #show enum: it => it + fakepar 

  #doc
]
