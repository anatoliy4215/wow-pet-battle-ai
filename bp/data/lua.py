"""Minimal parser for WoW SavedVariables files (Lua table literals)."""
import re

_TOKEN = re.compile(
    r'\s*(?:--[^\n]*\n\s*)*'
    r'(\{|\}|\[|\]|=|,|;|"(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|true|false|nil|[A-Za-z_]\w*)',
    re.S)
_ESC = re.compile(r'\\(\d{1,3}|.)', re.S)
_SIMPLE = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "'": "'"}


def _tokens(text):
    pos, n = 0, len(text)
    while pos < n:
        m = _TOKEN.match(text, pos)
        if not m:
            if text[pos:].strip() == "":
                return
            raise ValueError(f"bad lua at {pos}: {text[pos:pos + 40]!r}")
        pos = m.end()
        yield m.group(1)


def _unquote(s):
    body = s[1:-1]
    if "\\" not in body:
        return body

    def rep(m):
        g = m.group(1)
        if g.isdigit():
            return chr(int(g))
        return _SIMPLE.get(g, g)

    out = _ESC.sub(rep, body)
    try:  # decimal escapes are raw UTF-8 bytes
        return out.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return out


def _atom(tok):
    if tok.startswith('"'):
        return _unquote(tok)
    if tok == "true":
        return True
    if tok == "false":
        return False
    if tok == "nil":
        return None
    try:
        return int(tok)
    except ValueError:
        return float(tok)


def _value(it, tok):
    if tok != "{":
        return _atom(tok)
    arr, obj = [], {}
    tok = next(it)
    while tok != "}":
        if tok == "[":
            key = _value(it, next(it))
            assert next(it) == "]"
            assert next(it) == "="
            obj[key] = _value(it, next(it))
        else:
            arr.append(_value(it, tok))
        tok = next(it)
        if tok in (",", ";"):
            tok = next(it)
    if obj and arr:
        for i, v in enumerate(arr, 1):
            obj[i] = v
    if obj and all(isinstance(k, int) for k in obj) and sorted(obj) == list(range(1, len(obj) + 1)):
        return [obj[k] for k in sorted(obj)]
    return obj if obj else arr


def load_saved_variables(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    it = _tokens(text)
    out = {}
    for name in it:
        assert next(it) == "="
        out[name] = _value(it, next(it))
    return out
