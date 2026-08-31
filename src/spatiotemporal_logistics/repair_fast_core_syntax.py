#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


TARGET = Path(
    "src/spatiotemporal_logistics/"
    "01_build_fast_spatiotemporal_core.py"
)

if not TARGET.exists():
    raise SystemExit(
        f"[FAIL] arquivo não encontrado: {TARGET}"
    )


# =====================================================================
# BACKUP
# =====================================================================

stamp = datetime.now(
    timezone.utc
).strftime("%Y%m%dT%H%M%SZ")

backup = TARGET.with_name(
    TARGET.stem
    + f".before_syntax_repair_{stamp}"
    + TARGET.suffix
)

shutil.copy2(
    TARGET,
    backup
)

print("=" * 100)
print("FAST CORE — SAFE SYNTAX REPAIR")
print("=" * 100)
print()
print(f"[PASS] backup criado: {backup}")
print()


# =====================================================================
# LOAD
# =====================================================================

original = TARGET.read_text(
    encoding="utf-8"
)

lines = original.splitlines(
    keepends=True
)


def show_context(
    source_lines,
    lineno,
    radius=6
):
    start = max(
        1,
        lineno - radius
    )

    end = min(
        len(source_lines),
        lineno + radius
    )

    print()
    print("-" * 100)
    print(
        f"CONTEXTO {start}:{end}"
    )
    print("-" * 100)

    for number in range(
        start,
        end + 1
    ):
        marker = (
            ">>>"
            if number == lineno
            else "   "
        )

        content = source_lines[
            number - 1
        ].rstrip("\n")

        print(
            f"{marker} {number:5d} | "
            f"{content}"
        )

    print("-" * 100)
    print()


# =====================================================================
# COMPILE ORIGINAL IN MEMORY
# =====================================================================

candidate = list(lines)
repairs = []

MAX_SAFE_REPAIRS = 20


for attempt in range(
    MAX_SAFE_REPAIRS + 1
):

    source = "".join(
        candidate
    )

    try:
        compile(
            source,
            str(TARGET),
            "exec"
        )

        print(
            "[PASS] candidate compila "
            "corretamente em memória."
        )

        break

    except (
        IndentationError,
        SyntaxError,
    ) as exc:

        lineno = int(
            exc.lineno or 0
        )

        print(
            f"[INFO] erro de sintaxe detectado: "
            f"{type(exc).__name__}"
        )

        print(
            f"[INFO] linha={lineno}"
        )

        print(
            f"[INFO] mensagem={exc.msg}"
        )

        if (
            lineno <= 0
            or
            lineno > len(candidate)
        ):
            raise SystemExit(
                "[FAIL] número de linha inválido; "
                "arquivo original preservado."
            )

        show_context(
            candidate,
            lineno
        )

        # =============================================================
        # REPARO SEGURO:
        #
        # try:
        # try:
        #
        # com exatamente a mesma indentação.
        #
        # O primeiro try não possui suite e é sintaticamente impossível.
        # =============================================================

        if lineno >= 2:

            prev = candidate[
                lineno - 2
            ]

            curr = candidate[
                lineno - 1
            ]

            prev_stripped = (
                prev.strip()
            )

            curr_stripped = (
                curr.strip()
            )

            prev_indent = (
                len(prev)
                -
                len(
                    prev.lstrip(
                        " \t"
                    )
                )
            )

            curr_indent = (
                len(curr)
                -
                len(
                    curr.lstrip(
                        " \t"
                    )
                )
            )

            if (
                prev_stripped == "try:"
                and
                curr_stripped == "try:"
                and
                prev_indent
                ==
                curr_indent
            ):

                removed_line = (
                    lineno - 1
                )

                removed_text = (
                    candidate[
                        lineno - 2
                    ].rstrip("\n")
                )

                print(
                    "[SAFE REPAIR] "
                    "dois try consecutivos "
                    "no mesmo nível."
                )

                print(
                    "[SAFE REPAIR] "
                    f"removendo linha "
                    f"{removed_line}: "
                    f"{removed_text!r}"
                )

                repairs.append(
                    {
                        "type":
                            "DUPLICATE_TRY",

                        "line":
                            removed_line,

                        "text":
                            removed_text,
                    }
                )

                del candidate[
                    lineno - 2
                ]

                continue

        # =============================================================
        # NÃO TENTAR "ADIVINHAR" OUTROS ERROS
        # =============================================================

        print()
        print(
            "[FAIL] O erro seguinte não corresponde "
            "ao padrão seguro de try duplicado."
        )

        print(
            "[FAIL] Nenhuma alteração será gravada "
            "no arquivo principal."
        )

        print(
            f"[PASS] backup disponível: {backup}"
        )

        raise SystemExit(2)

else:

    raise SystemExit(
        "[FAIL] limite de reparos seguros excedido."
    )


# =====================================================================
# FINAL VALIDATION BEFORE WRITE
# =====================================================================

final_source = "".join(
    candidate
)

try:
    compile(
        final_source,
        str(TARGET),
        "exec"
    )

except Exception as exc:
    raise SystemExit(
        "[FAIL] candidate final não compilou: "
        f"{exc}"
    )


# =====================================================================
# WRITE ONLY AFTER SUCCESSFUL COMPILE
# =====================================================================

TARGET.write_text(
    final_source,
    encoding="utf-8"
)

print()
print("=" * 100)
print("REPAROS APLICADOS")
print("=" * 100)

if repairs:

    for repair in repairs:
        print(
            "[PASS] "
            f"{repair['type']} | "
            f"linha_original="
            f"{repair['line']}"
        )

else:

    print(
        "[INFO] nenhum reparo necessário."
    )


print()
print(
    f"[PASS] arquivo principal "
    f"gravado somente após compilação."
)

print(
    f"[PASS] backup preservado: "
    f"{backup}"
)

print(
    "RAW_MODIFIED = false"
)

print(
    "SYNTAX_REPAIR = PASS"
)
