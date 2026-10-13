import ast
import io
import os
import sys
import tokenize


def drop_comments(src):
    lines = src.splitlines(keepends=True)
    cols = {}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            ln, col = tok.start
            cols[ln] = min(cols.get(ln, 10 ** 9), col)
    out = []
    for i, line in enumerate(lines, start=1):
        if i in cols:
            head = line[: cols[i]].rstrip()
            out.append(head + "\n" if line.endswith("\n") and head else head)
        else:
            out.append(line)
    return "".join(out)


def is_docstring(node):
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def drop_docstrings(src):
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    drop = set()
    keep_pass = set()

    def handle(body, is_module):
        if not body or not is_docstring(body[0]):
            return
        first = body[0]
        end = first.end_lineno or first.lineno
        if len(body) == 1 and not is_module:
            keep_pass.add(first.lineno)
            return
        for ln in range(first.lineno, end + 1):
            drop.add(ln)
        if end + 1 <= len(lines) and lines[end].strip() == "":
            drop.add(end + 1)

    handle(tree.body, True)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            handle(node.body, False)

    out = []
    for i, line in enumerate(lines, start=1):
        if i in drop:
            continue
        if i in keep_pass:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(indent + "pass" + ("\n" if line.endswith("\n") else ""))
            continue
        out.append(line)
    return "".join(out)


def strip(path):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if not src.strip():
        return False
    try:
        cleaned = drop_docstrings(drop_comments(src))
        ast.parse(cleaned)
    except SyntaxError as exc:
        print("skip {}: {}".format(path, exc))
        return False
    if cleaned != src:
        with open(path, "w", encoding="utf-8") as f:
            f.write(cleaned)
        return True
    return False


def main(roots):
    targets = []
    for root in roots:
        if os.path.isfile(root):
            targets.append(root)
            continue
        for base, _, names in os.walk(root):
            if "__pycache__" in base:
                continue
            for name in names:
                if name.endswith(".py"):
                    targets.append(os.path.join(base, name))
    changed = [p for p in sorted(targets) if strip(p)]
    for p in changed:
        print("stripped", p)
    print("{} files changed".format(len(changed)))


if __name__ == "__main__":
    main(sys.argv[1:] or ["src", "scripts", "app.py"])
