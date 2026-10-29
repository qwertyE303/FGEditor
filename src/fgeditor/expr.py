"""Expression and domain validation for FGEditor.

Function expressions keep the v1 grammar.  Domain expressions use the v2.1
boolean grammar: ``and``/``&&``, ``or``/``||``, ``not``/``!``, parentheses and
chained comparisons.  Only syntax is checked; nothing is evaluated.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

COORDINATE_NOTE = (
    "Ballance里y轴为高度，且为左手系，因此定义："
    "Theta是位置矢量在xz平面投影和x的夹角（范围0~2pi），"
    "Phi是和位置矢量和xz平面的夹角（范围-0.5pi~0.5pi）"
)

CARTESIAN_VARS = frozenset({"x", "y", "z"})
SPHERICAL_VARS = frozenset({"r", "theta", "phi"})
DOMAIN_VARS = CARTESIAN_VARS | SPHERICAL_VARS
CONSTANTS = frozenset({"pi", "e"})


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    signature: str
    min_args: int
    max_args: int | None
    description: str
    example: str


FUNCTION_SPECS: dict[str, FunctionSpec] = {
    "sin": FunctionSpec("sin", "sin(x)", 1, 1, "正弦，弧度制", "sin(pi/2)"),
    "cos": FunctionSpec("cos", "cos(x)", 1, 1, "余弦，弧度制", "cos(0)"),
    "tan": FunctionSpec("tan", "tan(x)", 1, 1, "正切，弧度制", "tan(pi/4)"),
    "sec": FunctionSpec("sec", "sec(x)", 1, 1, "1/cos(x)，弧度制", "sec(0)"),
    "csc": FunctionSpec("csc", "csc(x)", 1, 1, "1/sin(x)，弧度制", "csc(pi/2)"),
    "cot": FunctionSpec("cot", "cot(x)", 1, 1, "1/tan(x)，弧度制", "cot(pi/4)"),
    "asin": FunctionSpec("asin", "asin(x)", 1, 1, "反正弦，返回弧度", "asin(1)"),
    "acos": FunctionSpec("acos", "acos(x)", 1, 1, "反余弦，返回弧度", "acos(1)"),
    "atan": FunctionSpec("atan", "atan(x)", 1, 1, "反正切，返回弧度", "atan(1)"),
    "atan2": FunctionSpec("atan2", "atan2(y,x)", 2, 2,
                          "标准 atan2，返回弧度；y 在前、x 在后", "atan2(y,x)"),
    "sinh": FunctionSpec("sinh", "sinh(x)", 1, 1, "双曲正弦", "sinh(1)"),
    "cosh": FunctionSpec("cosh", "cosh(x)", 1, 1, "双曲余弦", "cosh(1)"),
    "tanh": FunctionSpec("tanh", "tanh(x)", 1, 1, "双曲正切", "tanh(1)"),
    "asinh": FunctionSpec("asinh", "asinh(x)", 1, 1, "反双曲正弦", "asinh(1)"),
    "acosh": FunctionSpec("acosh", "acosh(x)", 1, 1, "反双曲余弦", "acosh(1)"),
    "atanh": FunctionSpec("atanh", "atanh(x)", 1, 1, "反双曲正切", "atanh(0.5)"),
    "ln": FunctionSpec("ln", "ln(x)", 1, 1, "自然对数", "ln(e)"),
    "log10": FunctionSpec("log10", "log10(x)", 1, 1, "以 10 为底的对数", "log10(100)"),
    "log": FunctionSpec("log", "log(a,b)", 2, 2,
                        "以 a 为底、b 为真数的对数；a 为底数", "log(2,8)"),
    "sqrt": FunctionSpec("sqrt", "sqrt(x)", 1, 1, "平方根", "sqrt(x^2+y^2)"),
    "cbrt": FunctionSpec("cbrt", "cbrt(x)", 1, 1, "立方根", "cbrt(8)"),
    "abs": FunctionSpec("abs", "abs(x)", 1, 1, "绝对值", "abs(-1)"),
    "sign": FunctionSpec("sign", "sign(x)", 1, 1, "符号：-1 / 0 / 1", "sign(x)"),
    "floor": FunctionSpec("floor", "floor(x)", 1, 1, "向下取整", "floor(1.9)"),
    "ceil": FunctionSpec("ceil", "ceil(x)", 1, 1, "向上取整", "ceil(1.1)"),
    "round": FunctionSpec("round", "round(x)", 1, 1, "四舍五入，.5 远离 0", "round(1.5)"),
    "trunc": FunctionSpec("trunc", "trunc(x)", 1, 1, "向 0 取整", "trunc(-1.9)"),
    "min": FunctionSpec("min", "min(a,b,...)", 2, None, "最小值，至少 2 个参数", "min(x,y)"),
    "max": FunctionSpec("max", "max(a,b,...)", 2, None, "最大值，至少 2 个参数", "max(x,y)"),
    "mod": FunctionSpec("mod", "mod(a,b)", 2, 2,
                        "取余；约定为 fmod，余数符号跟 a", "mod(5,2)"),
}


class ExprError(Exception):
    def __init__(self, message: str, pos: int, text: str):
        super().__init__(message)
        self.message = message
        self.pos = pos
        self.text = text

    def __str__(self) -> str:
        line = self.text.count("\n", 0, self.pos) + 1
        line_start = self.text.rfind("\n", 0, self.pos) + 1
        column = self.pos - line_start + 1
        return f"第 {line} 行第 {column} 列：{self.message}"


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    pos: int


_NUMBER_RE = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_OP_CHARS = set("+-*/^")


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char.isdigit() or (char == "." and index + 1 < length and text[index + 1].isdigit()):
            match = _NUMBER_RE.match(text, index)
            if not match:
                raise ExprError("无效的数字", index, text)
            tokens.append(Token("NUMBER", match.group(0), index))
            index = match.end()
            continue
        if char.isascii() and (char.isalpha() or char == "_"):
            match = _IDENT_RE.match(text, index)
            if not match:
                raise ExprError("无效的标识符", index, text)
            word = match.group(0)
            lower = word.lower()
            if lower in ("and", "or", "not"):
                tokens.append(Token(lower.upper(), word, index))
            else:
                tokens.append(Token("IDENT", word, index))
            index = match.end()
            continue
        if char in _OP_CHARS:
            tokens.append(Token("OP", char, index))
            index += 1
            continue
        if char == "(":
            tokens.append(Token("LPAREN", char, index))
            index += 1
            continue
        if char == ")":
            tokens.append(Token("RPAREN", char, index))
            index += 1
            continue
        if char == ",":
            tokens.append(Token("COMMA", char, index))
            index += 1
            continue
        if char == ";":
            raise ExprError("不支持分号", index, text)
        if char == "&":
            if text[index:index + 2] == "&&":
                tokens.append(Token("AND", "&&", index))
                index += 2
                continue
            raise ExprError("单独的 & 不合法；请写 &&", index, text)
        if char == "|":
            if text[index:index + 2] == "||":
                tokens.append(Token("OR", "||", index))
                index += 2
                continue
            raise ExprError("单独的 | 不合法；请写 ||", index, text)
        if char == "!":
            if text[index:index + 2] == "!=":
                tokens.append(Token("REL", "!=", index))
                index += 2
                continue
            tokens.append(Token("NOT", "!", index))
            index += 1
            continue
        if char in "=<>":
            pair = text[index:index + 2]
            if pair == "==":
                raise ExprError("不支持 ==；相等请写 =", index, text)
            if pair == "<>":
                raise ExprError("不支持 <>；不等请写 !=", index, text)
            if pair in ("<=", ">="):
                tokens.append(Token("REL", pair, index))
                index += 2
                continue
            tokens.append(Token("REL", char, index))
            index += 1
            continue
        raise ExprError("只支持英文半角字符", index, text)
    tokens.append(Token("EOF", "", length))
    return tokens


def _insert_implicit_multiplication(tokens: list[Token]) -> list[Token]:
    result: list[Token] = []
    for token in tokens:
        if result:
            left = result[-1]
            if _needs_implicit_multiplication(left, token):
                result.append(Token("OP", "*", token.pos))
        result.append(token)
    return result


def _needs_implicit_multiplication(left: Token, right: Token) -> bool:
    if left.kind in ("NUMBER", "RPAREN") and right.kind in ("NUMBER", "IDENT", "LPAREN"):
        return True
    if left.kind == "IDENT" and right.kind == "LPAREN":
        return left.value.lower() not in FUNCTION_SPECS
    return False


class _ArithmeticMixin:
    def __init__(self, tokens: list[Token], text: str, allowed_vars: frozenset[str]):
        self.tokens = tokens
        self.text = text
        self.allowed_vars = allowed_vars
        self.index = 0

    def peek(self) -> Token:
        return self.tokens[self.index]

    def advance(self) -> Token:
        token = self.peek()
        if token.kind != "EOF":
            self.index += 1
        return token

    def error(self, message: str, token: Token | None = None) -> ExprError:
        return ExprError(message, (token or self.peek()).pos, self.text)

    def expect(self, kind: str, display: str) -> Token:
        token = self.peek()
        if token.kind != kind:
            raise self.error(f"缺少 {display}", token)
        return self.advance()

    def parse_expression(self) -> None:
        self.parse_term()
        while self.peek().kind == "OP" and self.peek().value in ("+", "-"):
            self.advance()
            self.parse_term()

    def parse_term(self) -> None:
        self.parse_unary()
        while self.peek().kind == "OP" and self.peek().value in ("*", "/"):
            self.advance()
            self.parse_unary()

    def parse_unary(self) -> None:
        token = self.peek()
        if token.kind == "OP" and token.value in ("+", "-"):
            self.advance()
            self.parse_unary()
            return
        self.parse_power()

    def parse_power(self) -> None:
        self.parse_primary()
        if self.peek().kind == "OP" and self.peek().value == "^":
            self.advance()
            self.parse_unary()

    def parse_primary(self) -> None:
        token = self.peek()
        if token.kind == "NUMBER":
            self.advance()
            return
        if token.kind == "IDENT":
            name = token.value.lower()
            if name in FUNCTION_SPECS:
                self.advance()
                if self.peek().kind != "LPAREN":
                    raise self.error(f"函数 {name} 需要括号，例如 {name}(x)", token)
                self.advance()
                args = 0
                if self.peek().kind != "RPAREN":
                    self.parse_expression()
                    args += 1
                    while self.peek().kind == "COMMA":
                        self.advance()
                        self.parse_expression()
                        args += 1
                self.expect("RPAREN", "右括号 )")
                spec = FUNCTION_SPECS[name]
                if args < spec.min_args or (spec.max_args is not None and args > spec.max_args):
                    if spec.max_args is None:
                        expected = f"至少 {spec.min_args} 个"
                    elif spec.min_args == spec.max_args:
                        expected = f"{spec.min_args} 个"
                    else:
                        expected = f"{spec.min_args}~{spec.max_args} 个"
                    raise self.error(
                        f"函数 {name} 需要 {expected}参数，实际 {args} 个", token)
                return
            if name in CONSTANTS or name in self.allowed_vars:
                self.advance()
                return
            allowed = ", ".join(sorted(self.allowed_vars))
            raise self.error(
                f'未知标识符 "{token.value}"；当前列可用变量：{allowed}', token)
        if token.kind == "LPAREN":
            self.advance()
            self.parse_expression()
            self.expect("RPAREN", "右括号 )")
            return
        if token.kind == "EOF":
            raise self.error("表达式不完整", token)
        if token.kind == "REL":
            raise self.error("这里需要表达式，不能直接出现关系符", token)
        if token.kind == "COMMA":
            raise self.error("这里需要表达式，不能直接出现逗号", token)
        if token.kind == "RPAREN":
            raise self.error("这里需要表达式", token)
        if token.kind in ("AND", "OR", "NOT"):
            raise self.error("这里需要表达式，不能直接出现逻辑运算符", token)
        raise self.error("这里需要数字、变量、常量、函数或 (", token)


class _Parser(_ArithmeticMixin):
    pass


def _check_expression_tail(parser: _Parser) -> None:
    token = parser.peek()
    if token.kind == "EOF":
        return
    if token.kind == "REL":
        raise parser.error("函数表达式不能包含关系符（=、!=、<、<=、>、>=）", token)
    if token.kind == "COMMA":
        raise parser.error("函数表达式不能包含逗号（函数参数除外）", token)
    if token.kind in ("AND", "OR", "NOT"):
        raise parser.error("函数表达式不能包含逻辑运算符（and/or/not）", token)
    if token.kind == "RPAREN":
        raise parser.error("多余的右括号", token)
    if token.kind in ("IDENT", "NUMBER", "LPAREN"):
        raise parser.error("缺少运算符；两个标识符之间要写 *", token)
    raise parser.error("表达式后有多余内容", token)


def check_function(text: str, allowed_vars: frozenset[str] | set[str]) -> str | None:
    """Return a human-readable syntax error, or None if valid."""
    if not text.strip():
        return "函数不能为空"
    try:
        tokens = _insert_implicit_multiplication(tokenize(text))
        parser = _Parser(tokens, text, frozenset(v.lower() for v in allowed_vars))
        parser.parse_expression()
        _check_expression_tail(parser)
    except ExprError as exc:
        return str(exc)
    return None


class _DomainParser(_ArithmeticMixin):
    def parse_domain(self) -> None:
        self.parse_or()
        token = self.peek()
        if token.kind != "EOF":
            if token.kind == "COMMA":
                raise self.error("顶层逗号不再表示 AND；请使用 and", token)
            if token.kind == "RPAREN":
                raise self.error("多余的右括号", token)
            raise self.error("作用域表达式后有多余内容", token)

    def parse_or(self) -> None:
        self.parse_and()
        while self.peek().kind == "OR":
            self.advance()
            self.parse_and()

    def parse_and(self) -> None:
        self.parse_not()
        while self.peek().kind == "AND":
            self.advance()
            self.parse_not()

    def parse_not(self) -> None:
        if self.peek().kind == "NOT":
            self.advance()
            self.parse_not()
            return
        self.parse_atom()

    def parse_atom(self) -> None:
        if self.peek().kind == "LPAREN":
            saved = self.index
            try:
                self.parse_comparison()
                return
            except ExprError:
                self.index = saved
            self.advance()
            self.parse_or()
            self.expect("RPAREN", "右括号 )")
            return
        self.parse_comparison()

    def parse_comparison(self) -> None:
        self.parse_expression()
        if self.peek().kind != "REL":
            raise self.error("缺少关系符（=、!=、<、<=、>、>=）")
        self.advance()
        self.parse_expression()
        while self.peek().kind == "REL":
            self.advance()
            self.parse_expression()


def check_domain(text: str, allow_empty: bool = False) -> str | None:
    """Return a syntax error, or None if valid.

    CRegion domains cannot be empty.  ``allow_empty`` is available for callers
    that need to inspect an imported empty domain without flagging it.
    """
    if not text.strip():
        return None if allow_empty else "定义域不能为空"
    try:
        tokens = _insert_implicit_multiplication(tokenize(text))
        parser = _DomainParser(tokens, text, DOMAIN_VARS)
        parser.parse_domain()
    except ExprError as exc:
        return str(exc)
    return None


def help_html() -> str:
    """HTML for the '函数填写说明' dialog."""
    operator_rows = [
        ("+", "二元", "加", "1（最低）", "左"),
        ("-", "二元", "减", "1", "左"),
        ("*", "二元", "乘", "2", "左"),
        ("/", "二元", "除", "2", "左"),
        ("+", "一元", "正号", "3", "右"),
        ("-", "一元", "负号", "3", "右"),
        ("^", "二元", "幂", "4（最高）", "右"),
        ("( )", "括号", "改变优先级", "—", "—"),
    ]
    function_rows = []
    for spec in FUNCTION_SPECS.values():
        function_rows.append(
            "<tr>"
            f"<td><code>{html.escape(spec.signature)}</code></td>"
            f"<td>{html.escape(spec.description)}</td>"
            f"<td><code>{html.escape(spec.example)}</code></td>"
            "</tr>"
        )
    operator_html = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in operator_rows
    )
    constants_html = """
<tr><td><code>pi</code></td><td>常量</td><td>圆周率</td></tr>
<tr><td><code>e</code></td><td>常量</td><td>自然常数</td></tr>
<tr><td><code>x, y, z</code></td><td>变量</td><td>直角坐标；Domain 也允许</td></tr>
<tr><td><code>r, theta, phi</code></td><td>变量</td><td>球坐标；Domain 也允许</td></tr>
"""
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
body {{ font-family: "Microsoft YaHei UI"; font-size: 10pt; }}
h3 {{ margin: 14px 0 6px 0; }}
table {{ border-collapse: collapse; margin: 4px 0 10px 0; }}
th, td {{ border: 1px solid #999; padding: 3px 7px; vertical-align: top; }}
th {{ font-weight: bold; }}
code {{ font-family: Consolas, monospace; }}
.note {{ color: #444; }}
</style></head><body>
<p><b>{html.escape(COORDINATE_NOTE)}</b></p>

<h3>运算符</h3>
<table width="100%">
<tr><th>运算符</th><th>形式</th><th>含义</th><th>优先级</th><th>结合性</th></tr>
{operator_html}
</table>

<h3>常量与变量</h3>
<table width="100%">
<tr><th>名称</th><th>类型</th><th>说明</th></tr>
{constants_html}
</table>

<h3>函数</h3>
<table width="100%">
<tr><th>函数</th><th>说明</th><th>示例</th></tr>
{"".join(function_rows)}
</table>
</body></html>"""


def domain_help_html() -> str:
    """HTML for the '作用域填写说明' dialog."""
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
body {{ font-family: "Microsoft YaHei UI"; font-size: 10pt; }}
h3 {{ margin: 14px 0 6px 0; }}
table {{ border-collapse: collapse; margin: 4px 0 10px 0; }}
th, td {{ border: 1px solid #999; padding: 3px 7px; vertical-align: top; }}
th {{ font-weight: bold; }}
code {{ font-family: Consolas, monospace; }}
.note {{ color: #444; }}
</style></head><body>
<p><b>{html.escape(COORDINATE_NOTE)}</b></p>

<p>运算符、常量与变量、函数同场函数填写说明。</p>

<h3>逻辑运算符</h3>
<table width="100%">
<tr><th>运算符</th><th>含义</th><th>示例</th></tr>
<tr><td><code>and</code> / <code>&amp;&amp;</code></td><td>与</td><td><code>x&gt;0 and y&gt;0</code></td></tr>
<tr><td><code>or</code> / <code>||</code></td><td>或</td><td><code>x=1 or y=1</code></td></tr>
<tr><td><code>not</code> / <code>!</code></td><td>非</td><td><code>not (x=1)</code>、<code>!(x=1)</code></td></tr>
<tr><td><code>( )</code></td><td>改变优先级</td><td><code>(x=1 or y=1) and z&gt;0</code></td></tr>
</table>

<h3>优先级</h3>
<p>括号 &gt; 比较 &gt; <code>not</code> &gt; <code>and</code> &gt; <code>or</code>。</p>
</body></html>"""
