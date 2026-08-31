# MODEL 01 — Formal Functional-K Decision

## C3.6-C

The frozen temporal supervised experiment compared the baseline

\[
K=0
\]

against thirteen alternatives containing RAW-30D functional
principal-component scores.

For future month \(m\) and candidate \(K>0\),

\[
\Delta_{m,K}
=
AP_{m,K}
-
AP_{m,0}.
\]

The baseline contains only `ORDER_CORE_V1`.

The frozen experiment contained

\[
14\times17 = 238
\]

logistic-regression fits.

The independently recomputed mean Average Precision of the baseline was

\[
AP_{K=0}
=
0.0818409554.
\]

For every evaluated \(K>0\), the observed mean and median paired
Average-Precision differences were negative.

The conclusion remained negative under contiguous temporal deletion
sensitivity.

Circular block-bootstrap simultaneous inference identified no robust
functional challenger.

Importantly, simultaneous inference also does not support the stronger
claim that \(K=0\) is universally statistically superior to every
alternative.

Therefore the decision for the first logistic model is

\[
\boxed{K^\star_{MODEL\ 01}=0}.
\]

Thus

\[
\boxed{MODEL\ 01 = ORDER\_CORE\_V1\ only}.
\]

## Scope

This decision applies specifically to `MODEL_01_ORDER_LOGISTIC`.

It does not establish that temporal structure is universally
non-predictive, nor does it preclude nonlinear, geographic,
regime-dependent or external-context interactions.

## Next analytical branch

The next independent explanatory branch is the
`SPATIOTEMPORAL_LOGISTICS_AUDIT`, beginning with the joint study of

\[
\text{price}
+
\text{freight}
+
\text{delivery outcome}
+
\text{origin/destination}
+
\text{region}
+
\text{time}
+
\text{external context}.
\]

The branch will be kept analytically separate from the frozen MODEL 01
decision.
