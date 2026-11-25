"""order_002: CAE 再現実装（order_001）のゼミ発表スライドを生成する．

テンプレート `.ai/ai-pptx-kit/templates/research.pptx` をもとに，python-pptx と OOXML の直接編集で
スライドを作る（LibreOffice は使用しない）．数式は LaTeX -> MathML -> OMML に変換し，
PowerPoint のネイティブ数式 (a14:m) として挿入する．

実行（python-pptx を含む .ai/pptx-render/.venv_render を用いる）:
    .ai/pptx-render/.venv_render/bin/python programs/slides/build_cae_slides.py
出力: theis_cae_reproduction.pptx（プロジェクトルート）
"""
import copy
import os
import re
from xml.sax.saxutils import escape

import latex2mathml.converter
import mathml2omml
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Cm, Pt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
TEMPLATE = os.path.join(ROOT, ".ai", "ai-pptx-kit", "templates", "research.pptx")
OUTPUT = os.path.join(ROOT, "theis_cae_reproduction.pptx")
FIG = os.path.join(ROOT, "outputs", "ex001_kodak_cae", "figures", "slides")

NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "a14": "http://schemas.microsoft.com/office/drawing/2010/main",
      "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
      "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006"}
NSDECL = " ".join(f'xmlns:{k}="{v}"' for k, v in NS.items())

BLUE = RGBColor(0x00, 0x70, 0xC0)
LIGHT = RGBColor(0xDE, 0xEB, 0xF7)
RED = "FF0000"
LEFT, WIDTH = Cm(0.35), Cm(33.2)
BODY_TOP = Cm(2.4)

# ---------------------------------------------------------------------------
# 数式（LaTeX -> OMML）
# ---------------------------------------------------------------------------
_ITALIC = {**{chr(ord("a") + i): chr(0x1D44E + i) for i in range(26)},
           **{chr(ord("A") + i): chr(0x1D434 + i) for i in range(26)}, "h": "ℎ"}


def _italicize(text):
    """ASCII の英字を数学用イタリック文字に置き換える．

    Args:
        text (str): 文字列．
    Returns:
        str: 置き換え後の文字列．
    """
    return "".join(_ITALIC.get(ch, ch) for ch in text)


def omml(latex, size):
    """LaTeX を PowerPoint 用の OMML 文字列（m:oMath 要素）に変換する．

    Args:
        latex (str): LaTeX の数式．
        size (int): 文字サイズ [pt]．
    Returns:
        str: m:oMath 要素の XML 文字列（名前空間宣言なし）．
    """
    latex = latex.replace(r"\boldsymbol{\lambda}", "\U0001D6CC")
    xml = mathml2omml.convert(latex2mathml.converter.convert(latex))

    def run(match):
        sty, text = match.group(1), match.group(2)
        italic = sty == "i" and re.fullmatch(r"[A-Za-z]+", text) is not None
        text = _italicize(text) if italic else text
        rpr = (f'<a:rPr lang="en-US" sz="{size * 100}" b="0" i="{1 if italic else 0}">'
               '<a:latin typeface="Cambria Math" panose="02040503050406030204" pitchFamily="18" charset="0"/></a:rPr>')
        mrpr = '<m:rPr><m:sty m:val="p"/></m:rPr>' if not italic else ""
        return f"<m:r>{mrpr}{rpr}<m:t>{text}</m:t></m:r>"

    xml = re.sub(r'<m:r><m:rPr><m:sty m:val="(\w)"/></m:rPr><m:t>([^<]*)</m:t></m:r>', run, xml)
    xml = re.sub(r"<m:sup></m:sup>", "<m:sup><m:r><m:t></m:t></m:r></m:sup>", xml)
    return xml


# ---------------------------------------------------------------------------
# テキスト（段落の XML）
# ---------------------------------------------------------------------------
def _run(text, size, color=None, bold=False):
    """テキストの a:r 要素を作る．「^{...}」は上付き文字として扱う．

    Args:
        text (str): 文字列．
        size (int): 文字サイズ [pt]．
        color (str | None): 16 進の文字色．
        bold (bool): 太字にするか．
    Returns:
        str: a:r 要素の XML 文字列．
    """
    fill = f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>' if color else ""
    out = []
    for i, piece in enumerate(re.split(r"\^\{([^}]*)\}", text)):
        if not piece:
            continue
        base = ' baseline="30000"' if i % 2 == 1 else ""
        out.append(f'<a:r><a:rPr lang="ja-JP" altLang="en-US" sz="{size * 100}" b="{1 if bold else 0}"{base} dirty="0">'
                   f"{fill}</a:rPr><a:t>{escape(piece)}</a:t></a:r>")
    return "".join(out)


def paragraph(segments, size=28, align="l", indent=0.0, space_before=0, display=False):
    """段落の XML を作る．

    Args:
        segments (list | str): 文字列，または ("t"|"m"|"r"|"b", 内容) のリスト．
            "t": 通常の文字，"m": インライン数式 (LaTeX)，"r": 赤字，"b": 太字．
        size (int): 文字サイズ [pt]．
        align (str): "l"，"ctr"，"r"．
        indent (float): 左インデント [cm]．
        space_before (int): 段落前の間隔 [pt]．
        display (bool): True の場合，segments を 1 つの別行立て数式 (LaTeX) とする．
    Returns:
        tuple[str, bool]: (a:p 要素の XML 文字列, 数式を含むか)．
    """
    if isinstance(segments, str) and not display:
        segments = [("t", segments)]
    ppr = (f'<a:pPr marL="{int(Cm(indent))}" indent="0" algn="{align}">'
           f'<a:spcBef><a:spcPts val="{space_before * 100}"/></a:spcBef><a:buNone/></a:pPr>')
    if display:
        body = (f'<a14:m><m:oMathPara><m:oMathParaPr><m:jc m:val="centerGroup"/></m:oMathParaPr>'
                f"{omml(segments, size)}</m:oMathPara></a14:m>")
        ppr = ppr.replace(f'algn="{align}"', 'algn="ctr"')
        return f'<a:p>{ppr}{body}<a:endParaRPr lang="en-US" sz="{size * 100}"/></a:p>', True
    parts, has_math = [], False
    for kind, content in segments:
        if kind == "m":
            parts.append(f"<a14:m>{omml(content, size)}</a14:m>")
            has_math = True
        else:
            parts.append(_run(content, size, RED if kind == "r" else None, kind == "b"))
    return f'<a:p>{ppr}{"".join(parts)}<a:endParaRPr lang="ja-JP" sz="{size * 100}"/></a:p>', has_math


def textbox(slide, left, top, width, height, paragraphs, anchor="t", wrap=True):
    """段落（paragraph の戻り値）を持つテキストボックスを追加する．数式を含む場合は mc:AlternateContent で包む．

    Args:
        slide (pptx.slide.Slide): スライド．
        left, top, width, height (int): 位置と大きさ [EMU]．
        paragraphs (list[tuple[str, bool]]): paragraph() の戻り値のリスト．
        anchor (str): 縦方向の配置 ("t", "ctr", "b")．
        wrap (bool): 折り返すか．
    Returns:
        lxml.etree._Element: 追加した p:sp 要素．
    """
    shape = slide.shapes.add_textbox(left, top, width, height)
    tf = shape.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "ctr": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    for attr in ["lIns", "rIns", "tIns", "bIns"]:
        tf._txBody.bodyPr.set(attr, "36000")
    tx = tf._txBody
    for p in tx.findall(qn("a:p")):
        tx.remove(p)
    has_math = False
    for xml, math in paragraphs:
        tx.append(etree.fromstring(f"<wrap {NSDECL}>{xml}</wrap>")[0])
        has_math = has_math or math
    sp = shape._element
    if has_math:
        _wrap_math(sp)
    return sp


def _wrap_math(sp):
    """数式を含む p:sp を mc:AlternateContent（Choice: a14，Fallback: 数式を除いた図形）で包む．

    Args:
        sp (lxml.etree._Element): p:sp 要素．
    Returns:
        None
    """
    parent = sp.getparent()
    idx = parent.index(sp)
    alt = etree.SubElement(parent, f"{{{NS['mc']}}}AlternateContent", nsmap={"mc": NS["mc"]})
    parent.remove(alt)
    choice = etree.SubElement(alt, f"{{{NS['mc']}}}Choice", nsmap={"a14": NS["a14"]})
    choice.set("Requires", "a14")
    fallback = etree.SubElement(alt, f"{{{NS['mc']}}}Fallback")
    fb = copy.deepcopy(sp)
    for m in fb.iter(f"{{{NS['a14']}}}m"):
        m.getparent().remove(m)
    parent.remove(sp)
    choice.append(sp)
    fallback.append(fb)
    parent.insert(idx, alt)


def bullets(items, size=28):
    """「・」付きの箇条書きの段落リストを作る．

    Args:
        items (list): (階層, segments) のリスト．階層 0 は 28pt，1 は 24pt で字下げ．
        size (int): 階層 0 の文字サイズ．
    Returns:
        list[tuple[str, bool]]: 段落のリスト．
    """
    out = []
    for level, seg in items:
        seg = [("t", seg)] if isinstance(seg, str) else seg
        if level == 0:
            out.append(paragraph([("t", "・")] + seg, size, space_before=8))
        elif level == 1:
            out.append(paragraph(seg, size - 4, indent=1.0, space_before=2))
        elif level == 2:
            out.append(paragraph(seg, size, space_before=8))
        else:
            out.append(paragraph([("b", seg[0][1])], size, space_before=24))
    return out


# ---------------------------------------------------------------------------
# スライドの部品
# ---------------------------------------------------------------------------
class Deck:
    """テンプレートからスライドを作るためのヘルパー．"""

    def __init__(self):
        """テンプレートを読み込み，スライド番号の部品を保持してから既存スライドを削除する．"""
        self.prs = Presentation(TEMPLATE)
        self.sldnum = None
        for sh in self.prs.slides[1].shapes:
            if sh.is_placeholder and sh.placeholder_format.idx == 12:
                self.sldnum = copy.deepcopy(sh._element)
        sld_ids = self.prs.slides._sldIdLst
        for sid in list(sld_ids):
            self.prs.part.drop_rel(sid.get(qn("r:id")))
            sld_ids.remove(sid)
        self.layout_title = self.prs.slide_layouts[0]
        self.layout_body = self.prs.slide_layouts[1]

    def new(self, title):
        """タイトル付きのスライドを追加する（本文のプレースホルダーは削除する）．

        Args:
            title (str): スライドのタイトル．
        Returns:
            pptx.slide.Slide: 追加したスライド．
        """
        slide = self.prs.slides.add_slide(self.layout_body)
        for ph in list(slide.placeholders):
            if ph.placeholder_format.idx == 0:
                ph.text_frame.text = title
            else:
                ph._element.getparent().remove(ph._element)
        num = copy.deepcopy(self.sldnum)
        for t in num.iter(qn("a:t")):
            t.text = str(len(self.prs.slides))
        num.find(".//" + qn("p:cNvPr")).set("id", "9999")
        slide.shapes._spTree.append(num)
        return slide

    def save(self, path):
        """保存する．

        Args:
            path (str): 保存先．
        Returns:
            None
        """
        self.prs.save(path)


def body(slide, items, top=BODY_TOP, height=Cm(14.8), left=LEFT, width=WIDTH, size=28):
    """本文の箇条書きを配置する．

    Args:
        slide (pptx.slide.Slide): スライド．
        items (list): bullets() に渡す項目．
        top, height, left, width (int): 位置と大きさ [EMU]．
        size (int): 文字サイズ．
    Returns:
        None
    """
    textbox(slide, left, top, width, height, bullets(items, size))


def equation(slide, latex, top, number, size=24, height=Cm(1.9)):
    """別行立ての数式と式番号を配置する．

    Args:
        slide (pptx.slide.Slide): スライド．
        latex (str): LaTeX の数式．
        top (int): 上端 [EMU]．
        number (int): 式番号．
        size (int): 文字サイズ．
        height (int): 高さ [EMU]．
    Returns:
        None
    """
    textbox(slide, LEFT, top, Cm(30.0), height, [paragraph(latex, size, display=True)], anchor="ctr")
    textbox(slide, Cm(30.5), top, Cm(2.9), height, [paragraph(f"({number})", 24, align="r")], anchor="ctr")


def references(slide, refs):
    """左下に参考文献を配置する．

    Args:
        slide (pptx.slide.Slide): スライド．
        refs (list[str]): 参考文献の文字列．
    Returns:
        None
    """
    h = Cm(0.55 * len(refs) + 0.2)
    textbox(slide, LEFT, Cm(17.75) - h, WIDTH, h, [paragraph(r, 10) for r in refs], anchor="b")


def box(slide, left, top, width, height, text, fill=BLUE, font=RGBColor(0xFF, 0xFF, 0xFF), size=20, line=None):
    """角丸四角形のブロックを配置する．

    Args:
        slide (pptx.slide.Slide): スライド．
        left, top, width, height (int): 位置と大きさ [EMU]．
        text (str): 文字列（改行可）．
        fill (RGBColor | None): 塗りつぶしの色．None なら白．
        font (RGBColor): 文字色．
        size (int): 文字サイズ．
        line (RGBColor | None): 枠線の色．
    Returns:
        pptx.shapes.autoshape.Shape: 追加した図形．
    """
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shp.adjustments[0] = 0.15
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill if fill is not None else RGBColor(0xFF, 0xFF, 0xFF)
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(1.5)
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.margin_left = tf.margin_right = Cm(0.05)
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line_text in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line_text
        r.font.size = Pt(size)
        r.font.color.rgb = font
    return shp


def arrow(slide, x1, y1, x2, y2, color=RGBColor(0x40, 0x40, 0x40), width=2.0):
    """矢印を配置する．

    Args:
        slide (pptx.slide.Slide): スライド．
        x1, y1, x2, y2 (int): 始点と終点 [EMU]．
        color (RGBColor): 線の色．
        width (float): 線幅 [pt]．
    Returns:
        None
    """
    con = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    con.line.color.rgb = color
    con.line.width = Pt(width)
    ln = con.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")


def chain(slide, top, height, items, gap=Cm(0.55), left=LEFT, width=WIDTH, size=20):
    """横一列に並べたブロックを矢印でつなぐ．

    Args:
        slide (pptx.slide.Slide): スライド．
        top (int): 上端 [EMU]．
        height (int): 高さ [EMU]．
        items (list[tuple[str, RGBColor|None, RGBColor, float]]): (文字列, 塗り, 文字色, 相対幅)．
        gap (int): ブロックの間隔 [EMU]．
        left, width (int): 列全体の位置と幅 [EMU]．
        size (int): 文字サイズ．
    Returns:
        list[pptx.shapes.autoshape.Shape]: 追加したブロック．
    """
    total = sum(w for *_, w in items)
    unit = (width - gap * (len(items) - 1)) / total
    x, shapes = left, []
    for text, fill, font, w in items:
        line = RGBColor(0x40, 0x40, 0x40) if fill is None else None
        shapes.append(box(slide, int(x), top, int(unit * w), height, text, fill, font, size, line))
        x += unit * w + gap
    for a, b in zip(shapes[:-1], shapes[1:]):
        arrow(slide, a.left + a.width, top + height // 2, b.left, top + height // 2)
    return shapes


def table(slide, left, top, width, rows, col_widths=None, size=20, row_h=Cm(1.0), highlight=None):
    """テンプレートと同じ表スタイル（Medium Style 2 - Accent 1）で表を配置する．

    Args:
        slide (pptx.slide.Slide): スライド．
        left, top, width (int): 位置と幅 [EMU]．
        rows (list[list[str]]): 1 行目を見出しとする表の内容．
        col_widths (list[float] | None): 列幅の比．
        size (int): 文字サイズ．
        row_h (int): 行の高さ [EMU]．
        highlight (set[tuple[int, int]] | None): 赤字にするセル (行, 列)．
    Returns:
        pptx.table.Table: 追加した表．
    """
    n_r, n_c = len(rows), len(rows[0])
    frame = slide.shapes.add_table(n_r, n_c, left, top, width, row_h * n_r)
    tbl = frame.table
    ratios = col_widths or [1] * n_c
    for j, r in enumerate(ratios):
        tbl.columns[j].width = int(width * r / sum(ratios))
    for i, row in enumerate(rows):
        tbl.rows[i].height = row_h
        for j, text in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Cm(0.15)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.text = ""
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            for k, piece in enumerate(re.split(r"\^\{([^}]*)\}", text)):
                if not piece:
                    continue
                r = p.add_run()
                r.text = piece
                r.font.size = Pt(size)
                if k % 2 == 1:
                    r.font._rPr.set("baseline", "30000")
                if highlight and (i, j) in highlight:
                    r.font.color.rgb = RGBColor(0xFF, 0x00, 0x00)
                    r.font.bold = True
    return tbl


def picture(slide, name, left, top, width=None, height=None):
    """図を配置する．

    Args:
        slide (pptx.slide.Slide): スライド．
        name (str): outputs/.../figures/slides/ 内のファイル名．
        left, top (int): 位置 [EMU]．
        width, height (int | None): 大きさ [EMU]．
    Returns:
        pptx.shapes.picture.Picture: 追加した図．
    """
    return slide.shapes.add_picture(os.path.join(FIG, name), left, top, width, height)


# ---------------------------------------------------------------------------
# 参考文献
# ---------------------------------------------------------------------------
REF_THEIS = ("[1] L. Theis, W. Shi, A. Cunningham, and F. Huszár, “Lossy image compression with compressive "
             "autoencoders,” Proc. International Conference on Learning Representations (ICLR), 2017．")
REF_BALLE = ("[2] J. Ballé, V. Laparra, and E. P. Simoncelli, “End-to-end optimization of nonlinear transform codes "
             "for perceptual quality,” Proc. Picture Coding Symposium, 2016．")
REF_DIV2K = ("[3] E. Agustsson and R. Timofte, “NTIRE 2017 challenge on single image super-resolution: dataset and "
             "study,” Proc. IEEE CVPR Workshops, 2017．")
REF_TODERICI = ("[4] G. Toderici, D. Vincent, N. Johnston, S. J. Hwang, D. Minnen, J. Shor, and M. Covell, “Full resolution "
                "image compression with recurrent neural networks,” arXiv:1608.05148v1, 2016．")
REF_KODAK = "[5] Kodak Lossless True Color Image Suite，http://r0k.us/graphics/kodak/，参照: Nov. 18, 2026．"
REF_SUPP = ("[6] L. Theis, et al., Supplementary material，https://figshare.com/articles/supplementary_zip/4210152，"
            "参照: Nov. 18, 2026．")

GRAY = RGBColor(0x7F, 0x7F, 0x7F)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x00, 0x00, 0x00)
CONV = RGBColor(0x00, 0x70, 0xC0)
RES = RGBColor(0x9D, 0xC3, 0xE6)
SUB = RGBColor(0xED, 0x7D, 0x31)


# ---------------------------------------------------------------------------
# 各スライド
# ---------------------------------------------------------------------------
def s_title(d):
    """表紙．"""
    s = d.prs.slides.add_slide(d.layout_title)
    for ph in s.placeholders:
        idx = ph.placeholder_format.idx
        if idx == 0:
            ph.text_frame.text = "Compressive Autoencoderによる\n非可逆画像圧縮の再現実装"
            for para in ph.text_frame.paragraphs:
                for r in para.runs:
                    r.font.size = Pt(44)
        elif idx == 1:
            tf = ph.text_frame
            tf.text = "Theis et al.（ICLR 2017）のFigure 4の再現"
            p = tf.add_paragraph()
            p.text = "湘南工科大学"
            p = tf.add_paragraph()
            p.text = "（発表者名）"
    for ph in list(s.placeholders):
        if ph.placeholder_format.idx in (10, 11):
            ph._element.getparent().remove(ph._element)


def s_background(d):
    """研究背景．"""
    s = d.new("研究背景")
    body(s, [(0, "JPEGやJPEG 2000は，変換や量子化を人手で設計"),
             (0, "ニューラルネットワークで符号化器と復号器を学習すれば，"),
             (1, "画像の種類や評価指標に合わせて圧縮方式を最適化できる"),
             (0, [("t", "課題："), ("r", "量子化（丸め）と符号長が微分できない")]),
             (1, "勾配法で符号長と歪みのトレードオフを直接最適化できない"),
             (0, "Theis et al.[1]は微分の置き換えにより，この課題を解決"),
             (1, "JPEG 2000と同等の性能をKodak画像で報告")])
    references(s, [REF_THEIS])


def s_purpose(d):
    """本発表の目的．"""
    s = d.new("本発表の目的")
    body(s, [(0, "Theis et al.[1]のCompressive Autoencoder（CAE）をPyTorchで再現"),
             (0, "論文Figure 4（Kodak 24枚のレート・歪み曲線）を再現"),
             (1, "論文の値と同じ指標・同じ平均化の方法で比較"),
             (1, "横軸：1画素あたりのビット数（bpp），縦軸：PSNR，SSIM，MS-SSIM"),
             (0, "論文に記載のない仕様を明示し，再現できた点とできなかった点を整理")])
    references(s, [REF_THEIS])


def s_cae(d):
    """CAE の構成と目的関数．"""
    s = d.new("Compressive Autoencoder（CAE）の構成")
    top, h = Cm(2.7), Cm(2.4)
    chain(s, top, h, [("画像", None, BLACK, 0.8), ("Encoder\nf", CONV, WHITE, 1.1), ("丸め\n[・]", None, BLACK, 0.9),
                      ("符号化\n（Q）", RES, BLACK, 1.1), ("ビット列", None, BLACK, 1.0), ("復号\n（Q）", RES, BLACK, 1.1),
                      ("Decoder\ng", CONV, WHITE, 1.1), ("再構成\n画像", None, BLACK, 0.9)], size=22)
    textbox(s, LEFT, Cm(5.6), WIDTH, Cm(1.2),
            [paragraph("符号長と歪みの和を最小化する", 28)])
    equation(s, r"\mathcal{L} = -\log_2 Q\left([f(\mathbf{x})]\right) + \beta \, d\left(\mathbf{x}, g\left([f(\mathbf{x})]\right)\right)",
             Cm(6.9), 1, size=28, height=Cm(2.2))
    textbox(s, LEFT, Cm(9.5), WIDTH, Cm(7.0), [
        paragraph([("t", "ここで"), ("m", r"\mathbf{x}"), ("t", "は画像，"), ("m", "f"), ("t", "はEncoder，"), ("m", "g"),
                   ("t", "はDecoder，"), ("m", "[\\cdot]"), ("t", "は最近接整数への丸め，")], 24),
        paragraph([("m", "Q"), ("t", "は係数の離散確率モデル，"), ("m", "d"), ("t", "は歪み（平均二乗誤差），"),
                   ("m", r"\beta"), ("t", "は重みである")], 24),
        paragraph([("t", "・第1項はエントロピー符号化での"), ("b", "符号長"), ("t", "，第2項は"), ("b", "歪み")], 28, space_before=14),
        paragraph([("t", "・丸め"), ("m", "[\\cdot]"), ("t", "と"), ("m", "Q"), ("t", "が微分できないため，そのままでは勾配法で学習できない")], 28,
                  space_before=8)])
    references(s, [REF_THEIS])


def s_ste(d):
    """量子化の微分の置き換え．"""
    s = d.new("量子化の微分の置き換え")
    body(s, [(0, "順伝播では丸めをそのまま行い，逆伝播でのみ丸めの微分を置き換える")], height=Cm(1.6))
    equation(s, r"\frac{d}{dy}[y] := \frac{d}{dy} r(y) = 1, \quad r(y) = y", Cm(4.2), 2, size=28, height=Cm(2.4))
    textbox(s, LEFT, Cm(6.7), WIDTH, Cm(1.2), [paragraph(
        [("t", "ここで"), ("m", "y"), ("t", "はEncoderの出力，"), ("m", "r"), ("t", "は丸めの滑らかな近似（論文では恒等写像）である")], 24)])
    body(s, [(0, "Decoderの勾配は近似なしで計算できる"),
             (1, "加法的な一様雑音で量子化を置き換える方法[2]では，Decoderが雑音を含む入力で学習される"),
             (0, "出力の0〜255へのクリップも，微分を1と定義"),
             (1, "範囲外に出た画素にも，テスト時の誤差に比例した勾配が伝わる")], top=Cm(8.2), height=Cm(7.5))
    references(s, [REF_THEIS, REF_BALLE])


def s_rate(d):
    """符号長の推定．"""
    s = d.new("符号長の推定")
    body(s, [(0, [("t", "離散分布"), ("m", "Q"), ("t", "を連続密度"), ("m", "q"), ("t", "の積分で表し，Jensenの不等式で上界を求める")])],
         height=Cm(1.6))
    equation(s, r"-\log_2 Q(\mathbf{z}) \le \mathbb{E}_{\mathbf{u}}\left[-\log_2 q(\mathbf{z} + \mathbf{u})\right]",
             Cm(4.0), 3, size=28, height=Cm(2.0))
    textbox(s, LEFT, Cm(6.1), WIDTH, Cm(1.2), [paragraph(
        [("t", "ここで"), ("m", r"\mathbf{z}"), ("t", "は丸めた係数，"), ("m", r"\mathbf{u}"),
         ("t", "は[−0.5, 0.5)の一様乱数である（右辺は微分可能）")], 24)])
    body(s, [(0, [("m", "q"), ("t", "はチャネルごとに独立なガウス尺度混合（GSM，6成分）")])], top=Cm(7.4), height=Cm(1.6))
    equation(s, r"q(z_{kij}) = \sum_{s=1}^{6} \pi_{ks} \mathcal{N}\left(z_{kij}; 0, \sigma_{ks}^2\right)",
             Cm(9.0), 4, size=28, height=Cm(2.6))
    textbox(s, LEFT, Cm(11.7), WIDTH, Cm(2.6), [
        paragraph([("t", "ここで"), ("m", "k"), ("t", "はチャネル，"), ("m", "i, j"), ("t", "は空間位置，"), ("m", r"\pi_{ks}"),
                   ("t", "は混合比，"), ("m", r"\sigma_{ks}"), ("t", "は標準偏差である")], 24),
        paragraph("・混合比と精度は対数をとったパラメータで最適化する", 24, space_before=6)])
    references(s, [REF_THEIS])


def s_arch(d):
    """ネットワーク構造（論文 Figure 2）．"""
    s = d.new("ネットワーク構造（論文Figure 2）")
    textbox(s, LEFT, Cm(2.3), WIDTH, Cm(1.1), [paragraph("Encoder：ミラーパディングとvalid畳み込みで1/8に縮小", 24)])
    chain(s, Cm(3.5), Cm(2.6), [("画像", None, BLACK, 0.8), ("正規化\nミラー\nパディング", GRAY, WHITE, 1.15),
                                ("conv 5×5\n64ch，↓2", CONV, WHITE, 1.15), ("conv 5×5\n128ch，↓2", CONV, WHITE, 1.2),
                                ("Residual\nBlock ×3", RES, BLACK, 1.1), ("conv 5×5\nCch，↓2", CONV, WHITE, 1.1),
                                ("丸め", None, BLACK, 0.8), ("符号", None, BLACK, 0.8)])
    textbox(s, LEFT, Cm(6.9), WIDTH, Cm(1.1), [paragraph("Decoder：zero paddingの畳み込みとSub-pixel Convolutionで拡大", 24)])
    chain(s, Cm(8.1), Cm(2.6), [("符号", None, BLACK, 0.8), ("subpix 3×3\n512ch，↑2", SUB, WHITE, 1.25),
                                ("Residual\nBlock ×3", RES, BLACK, 1.1), ("subpix 3×3\n256ch，↑2", SUB, WHITE, 1.25),
                                ("subpix 3×3\n12ch，↑2", SUB, WHITE, 1.25), ("逆正規化\nクリップ", GRAY, WHITE, 1.1),
                                ("再構成\n画像", None, BLACK, 0.9)])
    body(s, [(0, [("t", "潜在チャネル数"), ("m", "C"), ("t", "：96（高・中ビットレート），64（低ビットレート）")]),
             (0, "Residual Block：conv 3×3（128ch）を2層とskip接続"),
             (0, "LeakyReLUの位置は論文Figure 2の矢印の塗り分けから判断")], top=Cm(11.2), height=Cm(5.0), size=24)
    references(s, [REF_THEIS])


def s_loss(d):
    """損失関数と 3 モデル．"""
    s = d.new("損失関数と3つのモデル")
    body(s, [(0, "論文Appendix A.4の形でレートと歪みの重みを指定する")], height=Cm(1.6))
    equation(s, r"\mathcal{L} = \frac{\alpha}{N}\left(-\ln q\left([f(\mathbf{x}) \circ \boldsymbol{\lambda}] + \mathbf{u}\right)\right)"
                r" + \frac{1-\alpha}{1000 M}\left\|\mathbf{x} - g\left([f(\mathbf{x}) \circ \boldsymbol{\lambda}] / \boldsymbol{\lambda}\right)\right\|^2",
             Cm(4.0), 5, size=24, height=Cm(2.6))
    textbox(s, LEFT, Cm(6.7), WIDTH, Cm(2.4), [
        paragraph([("t", "ここで"), ("m", r"\alpha"), ("t", "は重み，"), ("m", "N"), ("t", "は係数の数，"), ("m", "M"),
                   ("t", "は画素数の3倍，"), ("m", r"\boldsymbol{\lambda}"), ("t", "はスケール，")], 24),
        paragraph([("m", r"\circ"), ("t", "は要素ごとの積である（画素値は0〜255）")], 24)])
    table(s, Cm(6.6), Cm(10.6), Cm(20.0), [["モデル", "α", "潜在チャネル数"], ["高ビットレート", "0.01", "96"],
                                          ["中ビットレート", "0.05", "96"], ["低ビットレート", "0.2", "64"]],
          col_widths=[1.4, 1, 1.2], size=24, row_h=Cm(1.2))
    references(s, [REF_THEIS])


def s_training_tricks(d):
    """Incremental Training，スケール微調整，アンサンブル．"""
    s = d.new("学習と複数ビットレートへの対応")
    body(s, [(0, [("b", "Incremental Training"), ("t", "：2チャネルだけ有効にして学習を開始")]),
             (1, "改善が閾値未満になるたびに1チャネルずつ追加（マスクで無効化）"),
             (1, [("t", "全チャネルの有効化後に学習率を"), ("m", "10^{-4}"), ("t", "から"), ("m", "10^{-5}"), ("t", "に下げる")]),
             (0, [("b", "スケールの微調整"), ("t", "："), ("m", r"\boldsymbol{\lambda}"), ("t", "以外を固定し，別の"), ("m", r"\alpha"),
                  ("t", "で"), ("m", r"\boldsymbol{\lambda}"), ("t", "のみを最適化")]),
             (1, "1つのモデルで複数のビットレートに対応（10,000更新）"),
             (1, [("t", "隣り合う"), ("m", r"\boldsymbol{\lambda}"), ("t", "を補間してさらに細かく調整")]),
             (0, [("b", "アンサンブル"), ("t", "：画像とビットレートごとに歪み最小のモデルを選択")]),
             (1, "どのモデルを使ったかの情報として1 byteを符号長に加算")])
    references(s, [REF_THEIS])


def s_conditions(d):
    """実験条件．"""
    s = d.new("実験条件")
    rows = [["項目", "論文[1]", "本実験"],
            ["学習データ", "Flickrの画像434枚（非公開）", "DIV2K[3]の学習用800枚"],
            ["前処理", "1536×1536以下に縮小", "長辺1536 px以下に縮小"],
            ["パッチ・バッチ", "128×128，32枚", "128×128，32枚"],
            ["最適化・学習率", "Adam，0.0001→0.00001", "Adam，0.0001→0.00001"],
            ["更新回数", "最大1,000,000", "1,000,000（各約43時間）"],
            ["評価データ", "Kodak 24枚[5]", "Kodak 24枚[5]"],
            ["実装", "Theano，Lasagne", "PyTorch，RTX 3090"],
            ["エントロピー符号化", "rangecoder", "constriction（range coder）"]]
    table(s, LEFT, Cm(2.6), WIDTH, rows, col_widths=[1.3, 1.5, 1.5], size=24, row_h=Cm(1.4),
          highlight={(1, 2)})
    references(s, [REF_THEIS, REF_DIV2K, REF_KODAK])


def s_assumed(d):
    """論文に記載がなく推定した仕様．"""
    s = d.new("論文に記載がなく推定した仕様")
    body(s, [(2, "1．LeakyReLUの傾き0.01と重みの初期化（Lasagneの既定値）"),
             (2, "2．ミラーパディングの幅35 px（valid畳み込みで出力が1/8になる値）"),
             (2, "3．Incremental Trainingの判定"),
             (1, "1,000更新ごとに平均損失を比べ，相対改善率が1%未満なら1チャネル追加"),
             (2, [("t", "4．スケール微調整の目標"), ("m", r"\alpha"), ("t", "（0.002〜0.95の11通り）")]),
             (2, "5．GSMの初期値（標準偏差0.2〜20の対数等間隔）"),
             (2, "6．SSIMの設定（scikit-imageの既定値，RGB）")])


def s_extract(d):
    """論文の値の取得方法．"""
    s = d.new("論文の値はベクタPDFから数値として抽出")
    body(s, [(0, "論文の図と，著者公開の画像別の図[6]はベクタ形式のPDF"),
             (1, "折れ線の頂点座標を，目盛の位置から求めた写像で数値に変換"),
             (0, "画像別の値の平均（固定bppで補間）が論文Figure 4と一致"),
             (1, "論文の平均化の方法を確認し，本実験の曲線も同じ方法で計算")], height=Cm(7.2))
    rows = [["PSNR [dB]，1.0 bpp", "画像別の値から計算", "論文Figure 4"],
            ["CAE", "34.738", "34.730"], ["JPEG 2000", "35.076", "35.074"], ["JPEG", "32.768", "32.764"]]
    table(s, Cm(3.6), Cm(9.9), Cm(26.5), rows, col_widths=[1.2, 1.2, 1.0], size=24, row_h=Cm(1.15))
    references(s, [REF_SUPP])


def s_eval(d):
    """評価方法．"""
    s = d.new("評価方法：実際に符号化・復号して計測")
    body(s, [(0, "学習データの係数のヒストグラムとレンジコーダで符号化・復号"),
             (1, "実測の符号長は理想符号長の1.0003倍，復号した係数は全設定で完全に一致"),
             (0, "bpp = 符号長（モデル選択の1 byteを含む）/ 画素数 H×W"),
             (0, "JPEG（4:2:0，optimized）とJPEG 2000は同じ指標の実装で評価"),
             (1, "論文と同じくヘッダのビットは除外（JPEGのHuffmanテーブルは含む）"),
             (0, "PSNR，SSIM（scikit-image），MS-SSIM（Toderici et al.[4]の実装）"),
             (0, "Toderici et al.[4]は論文の図から抽出した値のみ使用")])
    references(s, [REF_TODERICI])


def s_train_result(d):
    """学習の推移．"""
    s = d.new("全チャネルの有効化後は性能がほぼ横ばい")
    picture(s, "training.png", Cm(0.9), Cm(2.5), width=Cm(32.0))
    body(s, [(1, "点線：全チャネルが有効になり学習率を下げた時点（16.0万，23.9万，25.6万更新）"),
             (1, "論文Figure 3Bでは約11.6万更新で学習率を下げており，本実験の方が遅い")],
         top=Cm(14.2), height=Cm(2.6))


def s_fig4(d):
    """Figure 4 の再現．"""
    s = d.new("Figure 4の再現：CAEは論文の曲線とほぼ一致")
    picture(s, "rd_curves.png", LEFT, Cm(2.3), width=WIDTH)
    body(s, [(1, "実線：本実験で計測した値（実測bpp），破線：論文の図から抽出した値")], top=Cm(15.0), height=Cm(1.3))
    references(s, [REF_THEIS])


def s_quant(d):
    """論文の CAE との定量比較．"""
    s = d.new("論文のCAEとの差はPSNRで約0.1 dB（1 bpp以下）")
    rows = [["指標", "曲線", "0.5 bpp", "1.0 bpp", "1.5 bpp", "2.0 bpp"],
            ["PSNR [dB]", "再現", "30.94", "34.83", "37.54", "39.72"],
            ["", "論文", "30.84", "34.73", "37.34", "39.28"],
            ["SSIM", "再現", "0.861", "0.927", "0.956", "0.971"],
            ["", "論文", "0.862", "0.927", "0.954", "0.967"],
            ["MS-SSIM", "再現", "0.960", "0.982", "0.990", "0.994"],
            ["", "論文", "0.956", "0.981", "0.989", "0.993"]]
    table(s, LEFT, Cm(2.6), WIDTH, rows, col_widths=[1.3, 0.9, 1, 1, 1, 1], size=24, row_h=Cm(1.2))
    body(s, [(0, "PSNRの差は1 bpp以下で+0.1 dB，高レート側で+0.2〜0.4 dB"),
             (0, "SSIMは1 bpp以下でほぼ同じ値"),
             (1, "本実験の評価環境はJPEGでもSSIMが約0.01高く，評価環境の差を含む可能性")],
         top=Cm(11.3), height=Cm(5.0))


def s_recon(d):
    """復元画像の例．"""
    import json
    with open(os.path.join(FIG, "recon_info.json")) as f:
        info = json.load(f)
    s = d.new("復元画像の例（kodim04，約0.37 bpp）")
    labels = [("original", "原画像"), ("cae", "CAE"), ("jpeg", "JPEG"), ("jpeg2000", "JPEG 2000")]
    w, gap = Cm(7.9), Cm(0.53)
    for i, (key, label) in enumerate(labels):
        x = LEFT + i * (w + gap)
        picture(s, f"recon_{key}.png", x, Cm(2.6), width=w)
        sub = "" if key == "original" else f"{info[key]['bpp']:.3f} bpp，{info[key]['psnr']:.2f} dB"
        textbox(s, x, Cm(8.65), w, Cm(1.9), [paragraph(label, 24, align="ctr"), paragraph(sub, 20, align="ctr")])
    body(s, [(0, "CAEはブロック状の歪みがなく滑らか，JPEG 2000は細部をより保持"),
             (1, "画像は，24枚のうちCAEとJPEGのPSNR差が中央値に近いものを選択"),
             (1, "JPEGはCAE以上のbppとなる最小の品質，JPEG 2000はCAEと同じbppを目標に符号化")],
         top=Cm(10.9), height=Cm(5.5))
    references(s, [REF_KODAK])


def s_jp2(d):
    """JPEG 2000 との比較．"""
    s = d.new("JPEG 2000に対する優劣は符号化器に依存")
    rows = [["PSNR [dB]", "0.5 bpp", "1.0 bpp", "2.0 bpp"],
            ["CAE（再現）", "30.94", "34.83", "39.72"],
            ["JPEG 2000（論文の値）", "31.46", "35.07", "39.40"],
            ["JPEG 2000（本評価，OpenJPEG）", "32.22", "36.02", "40.82"]]
    table(s, LEFT, Cm(2.6), WIDTH, rows, col_widths=[2.0, 1, 1, 1], size=24, row_h=Cm(1.2))
    body(s, [(0, "論文のJPEG 2000に対しては，PSNRは同程度でSSIMは上回る"),
             (1, "論文の結論を再現"),
             (0, [("t", "本評価のJPEG 2000に対しては，"), ("r", "PSNRで1.0〜1.5 dB劣る")]),
             (1, "SSIMはほぼ同等（2 bpp以上でわずかに上回る）"),
             (0, "論文で用いたJPEG 2000の符号化器は記載がない")], top=Cm(8.0), height=Cm(8.5))


def s_models(d):
    """各モデルの曲線（論文 Figure 9）．"""
    s = d.new("低ビットレートモデルの高レート化は再現できず")
    picture(s, "model_curves.png", Cm(0.9), Cm(2.4), width=Cm(32.0))
    body(s, [(1, [("t", "★："), ("m", r"\boldsymbol{\lambda}=1"), ("t", "（学習時の"), ("m", r"\alpha"),
                  ("t", "），実線：本実験の各レート設定，破線：論文Figure 9")])],
         top=Cm(15.2), height=Cm(1.3))


def s_discussion(d):
    """高レート化に失敗した原因の考察．"""
    s = d.new("考察：高レート化に失敗した原因")
    body(s, [(0, [("m", r"\boldsymbol{\lambda}"), ("t", "を大きくすると，Decoderの入力"),
                  ("m", r"[f(\mathbf{x}) \circ \boldsymbol{\lambda}] / \boldsymbol{\lambda}"), ("t", "が整数以外の値になる")]),
             (1, "学習中のDecoderは整数値の入力しか見ていない"),
             (0, [("t", "丸めの近似勾配は「"), ("m", r"\boldsymbol{\lambda}"), ("t", "を大きくすれば歪みが減る」方向を示す")]),
             (1, "実際の歪みは増加：歪み項 0.0514 → 0.0607（α=0.2のモデル，目標α=0.002）"),
             (0, "アンサンブルでは別のモデルが選ばれ，Figure 4への影響は小さい"),
             (0, [("t", "論文で同じ現象が起きなかった理由は"), ("r", "未特定")])])


def s_summary(d):
    """まとめと今後の課題．"""
    s = d.new("まとめ")
    body(s, [(0, "Theis et al.のCAEをPyTorchで再現し，3モデルを各100万更新学習"),
             (0, "Kodak 24枚での実際の符号化によりFigure 4を再現"),
             (1, "論文のCAEとのPSNR差は約0.1 dB（1 bpp以下）"),
             (0, "JPEG 2000に対する優劣は，JPEG 2000の符号化器に依存"),
             (0, "低ビットレートモデルのスケール微調整による高レート化は再現できず"),
             (3, "今後の課題"),
             (0, "高レート化に失敗した原因の調査（Decoderへの整数以外の入力）"),
             (0, "Incremental Trainingの閾値と学習データの影響の確認")])


def main():
    """全スライドを作成して保存する．

    Returns:
        None
    """
    d = Deck()
    for fn in [s_title, s_background, s_purpose, s_cae, s_ste, s_rate, s_arch, s_loss, s_training_tricks,
               s_conditions, s_assumed, s_extract, s_eval, s_train_result, s_fig4, s_quant, s_recon, s_jp2,
               s_models, s_discussion, s_summary]:
        fn(d)
    d.save(OUTPUT)
    print("saved:", OUTPUT, len(d.prs.slides), "slides")


if __name__ == "__main__":
    main()
