# Agent instructions

Read `catalog-info.yaml`, `engineering-policy.json`, `README.md`,
`docs/data.md` and `docs/runbook.md` before editing.

State a standards preflight with the `web-product` profile, pinned Engineering
Standards release, affected environment, data boundary, registered exceptions
and required evidence. Work in an isolated branch and deliver changes through a
Pull Request. Do not commit organizer data, images, embeddings, model weights,
credentials or password-protected source links.

Use identity-disjoint validation for model claims. Record dataset version,
checksum, split revision, configuration and metric outputs for every result.
Never use licence-plate information as a feature.

Before completion, run the repository's available checks, inspect the diff for
secrets and large artifacts, and state a standards postflight. Do not claim that
GitLab policy gates, a development environment or production acceptance passed
unless there is direct evidence.
