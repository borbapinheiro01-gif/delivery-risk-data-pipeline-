#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MODEL 01.0-C3.3-D2.1-B
SCIENTIFIC DOCUMENTATION — FUNCTIONAL REPRESENTATION DECISION
===============================================================================

Somente documentação.

NÃO executa:
- PCA;
- SVD;
- smoothing;
- target;
- classificação;
- seleção de K.

NÃO modifica:
- RAW;
- D2 original;
- decisão D2.1-A;
- documento-mestre.

A matemática é escrita como strings normais, NÃO como f-strings.
"""

from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import sys


PROJECT = (
    Path.home()
    / "workspace"
    / "Delivery_Risk_Intelligence"
)

DIR = (
    PROJECT
    / "reports"
    / "modeling"
    / "model_01_order_logistic"
)

DECISION = (
    DIR
    / "03s_functional_representation_decision.json"
)

VALIDATION = (
    DIR
    / "03r_d2_methodological_validation_v2.csv"
)

REPORT = (
    DIR
    / "03t_functional_representation_decision_report.txt"
)

OUT_MD = (
    PROJECT
    / "docs"
    / "FUNCTIONAL_REPRESENTATION_DECISION_D2.md"
)

OUT_VALIDATION = (
    DIR
    / "03v_d2_1b_documentation_validation.csv"
)

OUT_SUMMARY = (
    DIR
    / "03w_d2_1b_documentation_summary.json"
)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_csv(path):
    with path.open(
        encoding="utf-8",
        newline=""
    ) as f:
        return list(
            csv.DictReader(f)
        )


def add_check(
    rows,
    name,
    condition,
    observed,
    expected
):
    rows.append({
        "check": name,
        "status": "PASS" if condition else "FAIL",
        "observed": str(observed),
        "expected": str(expected)
    })


print()
print("=" * 108)
print("D2.1-B — SCIENTIFIC DOCUMENTATION")
print("=" * 108)


# =============================================================================
# 1. PREREQUISITES
# =============================================================================

for path in [
    DECISION,
    VALIDATION,
    REPORT
]:
    if not path.is_file():
        print("[FAIL] Ausente:", path)
        sys.exit(2)

    print("[PASS]", path.name)


decision_hash_before = sha256(
    DECISION
)

validation_hash_before = sha256(
    VALIDATION
)

report_hash_before = sha256(
    REPORT
)


decision = load_json(
    DECISION
)

validation = load_csv(
    VALIDATION
)


# =============================================================================
# 2. CHECK D2.1-A
# =============================================================================

checks = []


validation_failures = sum(
    row.get("status") == "FAIL"
    for row in validation
)


add_check(
    checks,
    "decision_pass",
    decision.get("status") == "PASS",
    decision.get("status"),
    "PASS"
)


add_check(
    checks,
    "validation_v2_zero_failures",
    validation_failures == 0,
    validation_failures,
    0
)


primary = decision.get(
    "primary_representation_for_next_stage",
    {}
)

challenger = decision.get(
    "smooth_challenger",
    {}
)

evidence = decision.get(
    "paired_evidence",
    {}
)


add_check(
    checks,
    "raw_30d_primary",
    (
        primary.get("representation") == "RAW_30D"
        and
        primary.get("status")
        ==
        "PRIMARY_FOR_NEXT_FUNCTIONAL_STAGE"
    ),
    primary,
    "RAW_30D / PRIMARY_FOR_NEXT_FUNCTIONAL_STAGE"
)


add_check(
    checks,
    "smooth_not_primary",
    (
        challenger.get("status")
        ==
        "REJECTED_AS_PRIMARY_UNSUPERVISED_PREPROCESSING"
    ),
    challenger.get("status"),
    "REJECTED_AS_PRIMARY_UNSUPERVISED_PREPROCESSING"
)


add_check(
    checks,
    "paired_comparisons",
    evidence.get("comparisons") == 34,
    evidence.get("comparisons"),
    34
)


add_check(
    checks,
    "raw_wins",
    evidence.get("raw_wins") == 34,
    evidence.get("raw_wins"),
    34
)


add_check(
    checks,
    "smooth_wins",
    evidence.get("smooth_wins") == 0,
    evidence.get("smooth_wins"),
    0
)


scope = decision.get(
    "scope_limits",
    {}
)


add_check(
    checks,
    "no_predictive_superiority_claim",
    scope.get("predictive_superiority_claimed") is False,
    scope.get("predictive_superiority_claimed"),
    False
)


add_check(
    checks,
    "final_k_not_selected",
    scope.get("final_k_selected") is False,
    scope.get("final_k_selected"),
    False
)


failures_before_doc = sum(
    row["status"] == "FAIL"
    for row in checks
)


if failures_before_doc:
    print("[STOP] D2.1-A inconsistente.")
    sys.exit(2)


# =============================================================================
# 3. EXTRACT CHANNEL RESULTS
# =============================================================================

channels = {
    x["channel"]: x
    for x in evidence.get(
        "channels",
        []
    )
}


if (
    "purchase_freight" not in channels
    or
    "purchase_volume" not in channels
):
    print("[FAIL] Canais ausentes.")
    sys.exit(2)


freight = channels[
    "purchase_freight"
]

volume = channels[
    "purchase_volume"
]


# =============================================================================
# 4. BUILD MARKDOWN
# =============================================================================
#
# IMPORTANTE:
# Não usamos f-string para matemática LaTeX.
# As linhas matemáticas abaixo são strings literais normais.
# Valores dinâmicos entram apenas em linhas simples.
# =============================================================================

lines = []


def w(text=""):
    lines.append(text)


w("# Functional Representation Decision — D2")
w()
w("## 1. Estado metodológico")
w()
w("**Decision ID:** `FUNCTIONAL_REPRESENTATION_D2_V1`")
w()
w("**Status:** `PASS`")
w()
w("**Representação primária para a próxima etapa:** `RAW-30D`")
w()
w(
    "**Smoothing challenger:** "
    "`SMOOTH-30D-EDF0.75` — "
    "`REJECTED_AS_PRIMARY_UNSUPERVISED_PREPROCESSING`"
)
w()
w(
    "Esta decisão é restrita à representação funcional não supervisionada "
    "e à fidelidade temporal da curva original. Ela não constitui prova "
    "de superioridade preditiva para a classificação de atraso."
)
w()


w("## 2. Contrato point-in-time")
w()
w("Para cada pedido i, o instante de previsão é:")
w()
w(r"\[")
w(r"t_{0,i}=T_i^{purchase}.")
w(r"\]")
w()
w("Um evento histórico j somente pode participar da representação se:")
w()
w(r"\[")
w(r"T_j^{purchase}<T_i^{purchase}.")
w(r"\]")
w()
w(
    "Assim, cada curva representa exclusivamente informação disponível "
    "antes do instante de previsão do pedido corrente."
)
w()


w("## 3. Representação funcional RAW-30D")
w()
w(
    "Para uma janela de 30 dias, cada pedido é associado a uma curva "
    "discreta:"
)
w()
w(r"\[")
w(r"H_i=\left(h_i(1),\ldots,h_i(30)\right).")
w(r"\]")
w()
w("A matriz centrada no conjunto de treino é:")
w()
w(r"\[")
w(r"X_R=H-\mathbf{1}\mu_R^T.")
w(r"\]")
w()
w("Sua decomposição SVD é:")
w()
w(r"\[")
w(r"X_R=U_R\Sigma_RV_R^T.")
w(r"\]")
w()
w("Com K componentes, a reconstrução é:")
w()
w(r"\[")
w(r"\widehat H_R")
w(r"=")
w(r"\mu_R+(H-\mu_R)V_{R,K}V_{R,K}^{T}.")
w(r"\]")
w()
w("O erro temporal utilizado é:")
w()
w(r"\[")
w(r"RE_R")
w(r"=")
w(
    r"\frac{\left\|H_{test}-\widehat H_R\right\|_F}"
    r"{\left\|H_{test}-\mu_R\right\|_F}."
)
w(r"\]")
w()


w("## 4. Representação suavizada avaliada")
w()
w("O challenger foi o suavizador de segunda diferença:")
w()
w(r"\[")
w(r"S_\lambda")
w(r"=")
w(r"\left(I+\lambda D_2^TD_2\right)^{-1}.")
w(r"\]")
w()
w("A curva suavizada é:")
w()
w(r"\[")
w(r"H^S=HS_\lambda.")
w(r"\]")
w()
w("A configuração avaliada foi:")
w()
w(r"\[")
w(r"EDF=22.5,\qquad EDF/30=0.75,")
w(r"\]")
w()
w("com")
w()
w(r"\[")
w(r"\lambda=0.0840662845049379.")
w(r"\]")
w()


w("## 5. Comparação temporal justa")
w()
w(
    "A PCA/SVD foi ajustada somente com pedidos anteriores ao mês de teste. "
    "O número K90 também foi aprendido somente no passado."
)
w()
w(
    "Além do erro interno no espaço suavizado, foi calculado o erro "
    "end-to-end contra a curva RAW futura:"
)
w()
w(r"\[")
w(r"RE_{S\rightarrow R}")
w(r"=")
w(
    r"\frac{\left\|H_{test}-\widehat H_S\right\|_F}"
    r"{\left\|H_{test}-\mu_R\right\|_F}."
)
w(r"\]")
w()
w("A diferença pareada por fold temporal é:")
w()
w(r"\[")
w(r"\Delta_m")
w(r"=")
w(r"RE_{S\rightarrow R,m}-RE_{R,m}.")
w(r"\]")
w()
w(
    "Quando Delta_m > 0, o smoothing possui maior erro end-to-end "
    "que a representação RAW naquele fold."
)
w()


w("## 6. Resultados empíricos")
w()
w("Foram avaliadas 34 comparações temporais pareadas:")
w()
w(r"\[")
w(r"34=17\times2,")
w(r"\]")
w()
w(
    "correspondentes a 17 períodos futuros para cada um dos dois canais."
)
w()


w("### 6.1 Purchase freight")
w()
w(
    "Testes temporais: "
    + str(freight["temporal_tests"])
)
w()
w(
    "RAW wins: "
    + str(freight["raw_wins"])
    + "/17"
)
w()
w(
    "SMOOTH wins: "
    + str(freight["smooth_wins"])
    + "/17"
)
w()
w(
    "Erro RAW futuro médio: "
    + format(
        freight["raw_future_error_mean"],
        ".10f"
    )
)
w()
w(
    "Erro SMOOTH->RAW futuro médio: "
    + format(
        freight[
            "smooth_end_to_end_raw_error_mean"
        ],
        ".10f"
    )
)
w()
w(
    "Delta médio: "
    + format(
        freight["delta_mean"],
        ".10f"
    )
)
w()
w(
    "Razão média SMOOTH/RAW: "
    + format(
        freight["ratio_mean"],
        ".10f"
    )
)
w()
w(
    "K90 mediano: RAW "
    + str(
        freight["raw_k90_median"]
    )
    + " -> SMOOTH "
    + str(
        freight["smooth_k90_median"]
    )
)
w()


w("### 6.2 Purchase volume")
w()
w(
    "Testes temporais: "
    + str(volume["temporal_tests"])
)
w()
w(
    "RAW wins: "
    + str(volume["raw_wins"])
    + "/17"
)
w()
w(
    "SMOOTH wins: "
    + str(volume["smooth_wins"])
    + "/17"
)
w()
w(
    "Erro RAW futuro médio: "
    + format(
        volume["raw_future_error_mean"],
        ".10f"
    )
)
w()
w(
    "Erro SMOOTH->RAW futuro médio: "
    + format(
        volume[
            "smooth_end_to_end_raw_error_mean"
        ],
        ".10f"
    )
)
w()
w(
    "Delta médio: "
    + format(
        volume["delta_mean"],
        ".10f"
    )
)
w()
w(
    "Razão média SMOOTH/RAW: "
    + format(
        volume["ratio_mean"],
        ".10f"
    )
)
w()
w(
    "K90 mediano: RAW "
    + str(
        volume["raw_k90_median"]
    )
    + " -> SMOOTH "
    + str(
        volume["smooth_k90_median"]
    )
)
w()


w("## 7. Resultado conjunto")
w()
w("Nos 34 pares:")
w()
w(r"\[")
w(r"\Delta_m>0")
w(r"\qquad")
w(r"\text{para }34/34\text{ comparações}.")
w(r"\]")
w()
w("Ao mesmo tempo:")
w()
w(r"\[")
w(r"K_{90}^{Smooth}<K_{90}^{RAW}")
w(r"\qquad")
w(r"\text{em }34/34\text{ comparações}.")
w(r"\]")
w()
w(
    "Portanto, o smoothing produziu maior compressão dimensional, "
    "mas menor fidelidade end-to-end em relação às curvas RAW futuras."
)
w()


w("## 8. Decisão")
w()
w(r"\[")
w(r"\boxed{\text{RAW-30D = PRIMARY FOR NEXT FUNCTIONAL STAGE}}")
w(r"\]")
w()
w(
    "`SMOOTH-30D-EDF0.75` é rejeitado como pré-processamento funcional "
    "não supervisionado primário."
)
w()
w(
    "A decisão não exclui a possibilidade de alguma transformação "
    "suavizada ser útil futuramente em um experimento supervisionado."
)
w()


w("## 9. Correção metodológica do D2 original")
w()
w(
    "O D2 original foi preservado com status FAIL por provenance. "
    "A única falha era a exigência de que os subconjuntos temporais "
    "futuros apresentassem valores negativos após smoothing."
)
w()
w(
    "D1.1 demonstrou negativos na população global auditada, enquanto "
    "os folds futuros do D2 apresentaram zero valores negativos. "
    "Essas observações não são contraditórias."
)
w()
w(
    "A validação metodológica V2 corrigiu a interpretação sem alterar "
    "os resultados originais do D2."
)
w()


w("## 10. Fundamentação teórica")
w()
w(
    "O smoothing por diferenças penalizadas é fundamentado como um "
    "problema de mínimos quadrados penalizados, no qual o parâmetro "
    "de regularização controla explicitamente o compromisso entre "
    "fidelidade e suavidade."
)
w()
w(
    "A literatura de Functional Data Analysis trata a análise de "
    "componentes principais funcionais como ferramenta de redução "
    "dimensional das curvas e distingue situações de observação densa, "
    "regular, esparsa e ruidosa."
)
w()
w(
    "No presente problema, as curvas são construídas deterministicamente "
    "em uma grade diária regular a partir dos eventos históricos PIT. "
    "Não foi estabelecido um modelo explícito de erro de medição que "
    "permitisse interpretar automaticamente a estrutura removida pelo "
    "smoothing como ruído."
)
w()
w(
    "A escolha entre RAW e smoothing foi, portanto, submetida a uma "
    "comparação temporal out-of-sample com múltiplas origens/períodos."
)
w()


w("## 11. Referências bibliográficas")
w()
w(
    "1. EILERS, P. H. C. A Perfect Smoother. "
    "Analytical Chemistry, v. 75, n. 14, p. 3631-3636, 2003. "
    "DOI: 10.1021/ac034173t."
)
w()
w(
    "2. RAMSAY, J. O.; SILVERMAN, B. W. "
    "Functional Data Analysis. 2. ed. "
    "New York: Springer, 2005. DOI: 10.1007/b98888."
)
w()
w(
    "3. HALL, P.; MÜLLER, H.-G.; WANG, J.-L. "
    "Properties of Principal Component Methods for Functional "
    "and Longitudinal Data Analysis. "
    "The Annals of Statistics, v. 34, n. 3, p. 1493-1517, 2006. "
    "DOI: 10.1214/009053606000000272."
)
w()
w(
    "4. WANG, J.-L.; CHIOU, J.-M.; MÜLLER, H.-G. "
    "Functional Data Analysis. "
    "Annual Review of Statistics and Its Application, "
    "v. 3, p. 257-295, 2016. "
    "DOI: 10.1146/annurev-statistics-041715-033624."
)
w()
w(
    "5. TASHMAN, L. J. "
    "Out-of-sample Tests of Forecasting Accuracy: "
    "An Analysis and Review. "
    "International Journal of Forecasting, "
    "v. 16, n. 4, p. 437-450, 2000. "
    "DOI: 10.1016/S0169-2070(00)00065-0."
)
w()
w(
    "6. BERGMEIR, C.; HYNDMAN, R. J.; KOO, B. "
    "A Note on the Validity of Cross-Validation for Evaluating "
    "Autoregressive Time Series Prediction. "
    "Computational Statistics & Data Analysis, "
    "v. 120, p. 70-83, 2018. "
    "DOI: 10.1016/j.csda.2017.11.003."
)
w()


w("## 12. Limites da conclusão")
w()
w("- Target utilizado nesta decisão: NÃO.")
w("- Superioridade preditiva afirmada: NÃO.")
w("- K final selecionado: NÃO.")
w("- Folds finais congelados: NÃO.")
w("- Classificador treinado: NÃO.")
w("- Silver criada: NÃO.")
w("- RAW modificado: NÃO.")
w()


w("## 13. Próxima etapa")
w()
w(
    "A representação que avança é `RAW-30D`. "
    "A próxima questão metodológica é definir uma política temporal "
    "para o número de componentes K, aprendida somente com informação "
    "disponível no passado."
)
w()


OUT_MD.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8"
)


# =============================================================================
# 5. VALIDATE DOCUMENT
# =============================================================================

text = OUT_MD.read_text(
    encoding="utf-8"
)


required_texts = [
    "RAW-30D",
    "PRIMARY FOR NEXT FUNCTIONAL STAGE",
    "34/34",
    "SMOOTH-30D-EDF0.75",
    "10.1021/ac034173t",
    "10.1007/b98888",
    "10.1214/009053606000000272",
    "10.1146/annurev-statistics-041715-033624",
    "10.1016/S0169-2070(00)00065-0",
    "10.1016/j.csda.2017.11.003",
    "Superioridade preditiva afirmada: NÃO",
    "K final selecionado: NÃO"
]


for token in required_texts:
    add_check(
        checks,
        "document_contains_" + token[:30],
        token in text,
        token in text,
        True
    )


# Original decision files must remain unchanged.
decision_hash_after = sha256(
    DECISION
)

validation_hash_after = sha256(
    VALIDATION
)

report_hash_after = sha256(
    REPORT
)


add_check(
    checks,
    "decision_json_preserved",
    decision_hash_before == decision_hash_after,
    decision_hash_after,
    decision_hash_before
)


add_check(
    checks,
    "validation_v2_preserved",
    validation_hash_before == validation_hash_after,
    validation_hash_after,
    validation_hash_before
)


add_check(
    checks,
    "decision_report_preserved",
    report_hash_before == report_hash_after,
    report_hash_after,
    report_hash_before
)


failures = sum(
    row["status"] == "FAIL"
    for row in checks
)


with OUT_VALIDATION.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "check",
            "status",
            "observed",
            "expected"
        ]
    )

    writer.writeheader()

    writer.writerows(
        checks
    )


summary = {
    "step":
        "MODEL_01_0_C3_3D2_1B_DOCUMENTATION",

    "status":
        "PASS"
        if failures == 0
        else "FAIL",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "source_decision":
        "03s_functional_representation_decision.json",

    "source_decision_status":
        decision.get("status"),

    "primary_representation":
        "RAW_30D",

    "paired_comparisons":
        evidence.get("comparisons"),

    "raw_wins":
        evidence.get("raw_wins"),

    "smooth_wins":
        evidence.get("smooth_wins"),

    "documentation_artifact":
        str(
            OUT_MD.relative_to(PROJECT)
        ),

    "master_document_modified":
        False,

    "pca_executed":
        False,

    "svd_executed":
        False,

    "smoothing_executed":
        False,

    "target_used":
        False,

    "model_trained":
        False,

    "silver_created":
        False,

    "raw_modified":
        False,

    "validation_failures":
        failures
}


OUT_SUMMARY.write_text(
    json.dumps(
        summary,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print()
print("=" * 108)
print("VALIDATION")
print("=" * 108)

for row in checks:
    print(
        "["
        + row["status"]
        + "] "
        + row["check"]
    )


print()
print("FAILURES =", failures)

print()
print("=" * 108)
print("RESULTADO D2.1-B")
print("=" * 108)

print("Decision source        :", decision.get("status"))
print("Primary representation : RAW_30D")
print("RAW wins               :", evidence.get("raw_wins"), "/ 34")
print("SMOOTH wins            :", evidence.get("smooth_wins"), "/ 34")
print("Documento criado       :", OUT_MD)
print()
print("Documento-mestre       : NÃO ALTERADO")
print("PCA                     : NÃO")
print("SVD                     : NÃO")
print("Smoothing               : NÃO")
print("Target                  : NÃO")
print("Modelo                  : NÃO")
print("Silver                  : NÃO")
print("RAW                     : INTACTO")


if failures:
    sys.exit(2)


print()
print("[PASS] D2.1-B documentação científica validada.")
print("[PASS] Parar antes da seleção temporal de K.")
