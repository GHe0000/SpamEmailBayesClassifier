#import "./module.typ": *

// 自定义数学简写
#show: shorthands.with(
  ($!=>$, $arrow.r.double.not$)
)

// 修改证毕符号
#show: thmrules.with(qed-symbol: $square$)

// PDF 文档元信息设置
#set document(
  title: [概率论与数理统计扩展任务8],
  author: ("Guotao He",),
  date: datetime.today()
)

// 页面设置
#set page(
  paper: "a4",
  margin: (top: 2.54cm,left:3.18cm,right:3.18cm,bottom:2.54cm)
)


// 字体设置
// 设置主字体. AMS uses the LaTeX font.
#set text(font: ("New Computer Modern","SimSun"),region: "cn",lang: "zh",normalsize)
#show emph: text.with(font: ("Times New Roman", "KaiTi"))
#show strong: text.with(font: ("New Computer Modern","SimHei"))

// 水印
//#set page(background: rotate(-45deg,
//  text(150pt, fill: rgb("E0E0E0"))[
//    DRAFT
//  ]
//))

// 设置行间距和段间距
#set par(justify: true, leading: 0.88em)
#set par(spacing: 1.6em)

// 自定义 Section 样式
#show heading.where(level:1): it => {
  counter(math.equation).update(0) // 更新编号
  set align(center)
  set text(LARGE)
  v(15pt)
  if it.numbering != none {
  strong([]+counter(heading).display(it.numbering))
}
  strong(it.body)
  // v(normalsize)
}

// 自定义 Subsection 样式
#show heading.where(level:2): it => {
  set align(center)
  set text(Large)
  set par(first-line-indent: 0em)
  v(8pt)
  if it.numbering != none {
  strong(counter(heading).display(it.numbering))
}
  strong(it.body)
  v(6pt)
}

#show heading.where(level:3): it => {
  set par(first-line-indent: 0em)
  set text(large)
  if it.numbering != none {
  strong(counter(heading).display(it.numbering))
}
  strong(it.body)
}
// 自定义目录
#set outline.entry(fill: repeat([. #h(0.25em)]))
#show outline.entry.where(level: 1): set outline.entry(fill: [ ])
#show outline.entry.where(level: 1): set block(above: 2.2em)
#show outline.entry.where(level: 1): it => link(
  it.element.location(),
  it.indented(it.prefix(), strong(it.inner())),
)

//#show outline.entry.where(level: 1): it => {
//  v(1em)
//  parbreak()
//  let res = link(it.element.location(), 
//    if it.element.numbering != none {
//      numbering(it.element.numbering, counter(heading).at(it.element.location()).at(0))
//    } + h(0.4em) + it.element.body
//    )
//  res += box(width: 1fr, ) 
//  res += link(it.element.location(), it.indented(it.prefix(), it.inner()))
//  strong(res)
//}

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
  numbering: "(1.1-1)",
  prefix: "eq:",
  only-labeled: true,  // 只编号有 label 的公式
  unnumbered-label: "", // 不编号的 label
)

#show figure.where(kind: image).or(figure.where(kind: table)): i-figured.show-figure.with(
  level: 2,
  zero-fill: true,
  leading-zero: true,
  numbering: "1.1-1",
)

#set heading(numbering: "1.1  ")

// 查找当前页或者前一个 Heading
#let numberingH(c)={
  if c.numbering == none {
    return none
  } else {
    return numbering(c.numbering,..counter(heading).at(c.location()))
    }
}

#let currentH(level: 1, supplement: [])={
  let elems = query(selector(heading.where(level: level)).after(here()))
  if elems.len() != 0 and elems.first().location().page() == here().page() {
    return [#supplement #numberingH(elems.first()) #elems.first().body] 
  } else {
    elems = query(selector(heading.where(level: level)).before(here()))
    if elems.len() != 0 {
      return [#supplement #numberingH(elems.last()) #elems.last().body] 
    }
  }
  return ""
}
// 自定义页眉
//#set page(header: context{
//  if here().page() != 1 {
//    if calc.rem(here().page(), 2) == 0 [                        // 偶数页
//      #align(left, text(currentH(level: 1, supplement: [Chapter]), size: normalsize))
//    ] else [                                                    // 奇数页
//      #align(right, text(currentH(level: 2), size: normalsize))
//    ]
//}
//})

// 自定义页脚，奇数页在右侧，偶数页在左侧
#set page(footer: context{
  if page_number_display.get() != none {
    set text(normalsize)
    if calc.odd(counter(page).get().at(0)) {
      align(right, counter(page).display(page_number_display.get()))
    }
    else {align(left, counter(page).display(page_number_display.get()))}
  }
})


// 正文内容
// 封面
#[
  #set par(first-line-indent: 0em)
  #set page(footer: none)
  #v(1fr)
  #align(center, {
      text(size: huge,[概率论与数理统计])
      v(25pt, weak: true)
      text(size: large, [扩展任务 8 试验报告])
      v(25pt, weak: true)
      text(size: large, [GHe])
      v(25pt, weak: true)
      text(size: large, [GitHub：#link("https://github.com/GHe0000/SpamEmailBayesClassifier")])
      parbreak()
      text(size: large, [Build: ] + datetime.today().display())
      parbreak()
      text(size: large, [Typst Version: ] + str(sys.version))
      
    })
  #v(1fr, weak: true)
]
//#pagebreak()
// 目录
#counter(page).update(1)
#page_number_display.update("i")


#[
#set par(first-line-indent: 0em)

#v(1fr)
本文使用了如下 AI 模型：

- DeepSeek-R1：各种提问，各种解释，辅助编写文档
- FittenCode：代码编写辅助 AI

本文所有代码均在 GitHub 仓库公开，仓库链接：#link("https://github.com/GHe0000/SpamEmailBayesClassifier") 中，包含：

- 原始数据预处理代码
- 模型训练和推理代码
- 模型分析和可视化代码
- 部分经过预处理的数据集
- 用于导出本 PDF 的 Typst 源文件以及自定义的模板

由于文件较多，代码量较长，因此本文并未放出全部代码，仅对关键部分进行了简要阐述.
#v(1fr)
]

#pagebreak()



#outline(indent: auto)
//#outline(indent: auto, fill: repeat([. #h(0.25em)]))
#pagebreak()

// 正式内容
#set par(first-line-indent: 2em)
#page_number_display.update("1")
#counter(page).update(1)

#include "Content.typ"
