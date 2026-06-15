━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 📦 Plugin update: v1.14.5 → v1.60.2 (minor)
**Security & CI hardening** - bring the SAST/dependency-audit gates the project lacked, and close the one untrusted-link gap in the injection scanner, *before* the detection engine lands its large new surface. Sourced from a reconciliation against the upstream framework's recent releases (`.planning/audits/UPSTREAM-GSD-CORE-DIFF-2026-06-13.md`).

### Added

- **CodeQL / SAST workflow** (`.github/workflows/codeql.yml`) - `javascript-typescript` with the `security-extended` query suite, on push / 
 Install: /gdd:update   Dismiss: /gdd:check-update --dismiss
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
