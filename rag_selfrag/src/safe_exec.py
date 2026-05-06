"""
safe_exec — execução restrita de código Python gerado por LLM.

Duas camadas de proteção:
  1. AST: detecta e bloqueia Import/ImportFrom antes da execução.
  2. Builtins: substitui __builtins__ por whitelist explícita.
"""
import ast

from src.logger import get_logger

log = get_logger(__name__)

_SAFE_BUILTINS: dict = {
    "bool": bool, "int": int, "float": float, "str": str,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "abs": abs, "round": round, "sum": sum, "min": min, "max": max,
    "len": len, "range": range, "enumerate": enumerate,
    "zip": zip, "sorted": sorted, "reversed": reversed,
    "format": format,
    "isinstance": isinstance, "type": type, "hasattr": hasattr,
    "vars": vars,
    "print": lambda *a, **kw: None,
}

_FORBIDDEN = (ast.Import, ast.ImportFrom)

_PREAPPROVED_MODULES = frozenset({"pandas"})


def safe_exec(code: str, ns: dict) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Sintaxe inválida no código gerado: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN):
            if isinstance(node, ast.Import):
                imported = {alias.name.split(".")[0] for alias in node.names}
                if imported <= _PREAPPROVED_MODULES:
                    continue
            stmt = ast.unparse(node) if hasattr(ast, "unparse") else ast.dump(node)
            log.warning(
                "Codigo gerado pelo LLM bloqueado por import proibido: '%s'", stmt,
                extra={"event": "safe_exec_blocked"},
            )
            raise ValueError(f"Import não permitido no código gerado: '{stmt}'")

    exec_ns = {"__builtins__": _SAFE_BUILTINS, **ns}
    try:
        exec(code, exec_ns)  # noqa: S102
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    ns.update({
        k: v
        for k, v in exec_ns.items()
        if k != "__builtins__" and not k.startswith("__")
    })