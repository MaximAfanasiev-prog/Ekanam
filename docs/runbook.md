# Runbook

## Current state

Bootstrap only. There is no deployed environment, health endpoint, model release
or production data path.

## Local data

Set `LCT_DATA_DIR` to a directory outside the repository. Confirm provenance,
access restriction, byte size and SHA-256 before use. Never print credentials or
the protected source password in logs.

## Required runtime contract

Before the first runnable release, document:

- reproducible training and validation commands;
- model/checkpoint provenance and size;
- API and browser-demo startup;
- health check and dependency checks;
- offline Docker Compose startup;
- latency and memory measurement;
- rollback to the previous immutable image digest;
- owner and escalation route.

## Delivery boundary

No direct deployment is authorized. Development and production environments are
not yet registered. Add them to `catalog-info.yaml` only after their namespaces,
data boundaries, health identifiers and delivery path are verified.
