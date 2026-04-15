"""
safe_exec — execução restrita de código Python gerado por LLM.

Duas camadas de proteção:
  1. AST: detecta e bloqueia Import/ImportFrom antes da execução.
  2. Builtins: substitui __builtins__ por whitelist explícita —
     sem __import__, open, exec, eval, compile ou acesso ao sistema.
"""
import ast

from src.logger import get_logger

log = get_logger(__name__)

# Apenas funções seguras da stdlib, suficientes para código pandas + formatação
_SAFE_BUILTINS: dict = {
    # Tipos primitivos
    "bool": bool, "int": int, "float": float, "str": str,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    # Aritmética e comparação
    "abs": abs, "round": round, "sum": sum, "min": min, "max": max,
    # Iteração
    "len": len, "range": range, "enumerate": enumerate,
    "zip": zip, "sorted": sorted, "reversed": reversed,
    # Strings e formatação
    "format": format,
    # Inspeção de tipo (usada por pandas internamente)
    "isinstance": isinstance, "type": type, "hasattr": hasattr,
    # print bloqueado silenciosamente (LLM às vezes gera, não deve causar erro)
    "print": lambda *a, **kw: None,
}

_FORBIDDEN = (ast.Import, ast.ImportFrom)


def safe_exec(code: str, ns: dict) -> None:
    """
    Executa `code` com namespace restrito.

    Parâmetros
    ----------
    code : str
        Código Python gerado pelo LLM.
    ns : dict
        Namespace de entrada (ex: {"pd": pd, "df": df}).
        Atualizado in-place com as variáveis definidas pelo código.

    Raises
    ------
    ValueError
        Se o código contiver imports ou sintaxe inválida.
    RuntimeError
        Se a execução lançar qualquer exceção.
    """
    # Camada 1 — validação AST (antes de qualquer execução)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Sintaxe inválida no código gerado: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN):
            stmt = ast.unparse(node) if hasattr(ast, "unparse") else ast.dump(node)
            log.warning(
                "Codigo gerado pelo LLM bloqueado por import proibido: '%s'", stmt,
                extra={"event": "safe_exec_blocked"},
            )
            raise ValueError(f"Import não permitido no código gerado: '{stmt}'")

    # Camada 2 — execução com builtins restritos
    exec_ns = {"__builtins__": _SAFE_BUILTINS, **ns}
    try:
        exec(code, exec_ns)  # noqa: S102
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    # Propaga variáveis definidas pelo código de volta para ns
    ns.update({
        k: v
        for k, v in exec_ns.items()
        if k != "__builtins__" and not k.startswith("__")
    })
