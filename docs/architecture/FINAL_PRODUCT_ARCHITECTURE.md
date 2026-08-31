# Delivery Risk & Shipping Friction Intelligence Platform

## End-to-End Architecture

DATA FOUNDATION
→ LOGISTICS INTELLIGENCE
→ PREDICTIVE AI
→ DECISION ENGINE
→ MLOps / CLOUD

## Scientific Core

Olist
→ Order x Seller
→ Order Level
→ Municipal / External Context
→ Spatiotemporal Core
→ Expected Freight OOT
→ Shipping Friction

## Predictive AI

Late Risk production benchmark:

M0 = Core Point-in-Time features

M1 = Core Point-in-Time features + Point-in-Time-safe Shipping Intelligence

Primary metric: PR-AUC.

Operational metric: Recall@Top-K.

## ETA

Deterministic ETA followed by probabilistic P50 / P80 / P95.

## Decision Engine

Prediction
→ Risk Ranking
→ Operational Capacity
→ Expected Intervention Value
→ Recommended Action

## Serving

MLflow
→ Model Registry
→ FastAPI
→ Docker
→ AWS

## Scientific Guardrails

prediction != causality

anomaly != proven inefficiency

high freight != logistics failure

post-outcome data != purchase-time predictor

diagnostic intelligence != predictive feature
