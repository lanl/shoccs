# Continuous integration (`.github/workflows/`)

## Workflows

- **`ci.yml`** — runs on pull requests and pushes to `main`.
  - `cpp` job: builds and tests inside the prebuilt dependency image `ghcr.io/<owner>/shoccs-ci`. The owner is derived from `github.repository_owner`, so the image is `lanl/shoccs-ci` upstream and `pbrady/shoccs-ci` on the fork; the job is visibly skipped if that image has not been published yet. The gating ctest run excludes `t-laplacian` (the known failure, `docs/CLEANUP_PLAN.md` §0a), which runs separately as a non-gating step. Benchmarks are compile-checked, never executed.
  - `python` job: runs the stencil_gen suite with the slow tests deselected, via uv (`UV_PYTHON` pins the interpreter).
- **`devcontainer-image.yml`** — builds the lean `ci` stage of `.devcontainer/Dockerfile` and pushes it to GHCR. Triggers: pushes to `main` touching `.devcontainer/` or `spack_repo/`, a weekly cron, or manual dispatch. Dispatch it after changing the dependency stack; a cold build takes hours because spack builds everything from source.
- **`python-extended.yml`** — weekly or manual run of the stencil_gen suite including the `--run-slow` tests. Full-resolution sweeps stay out of CI.

## Notes

- The workflow files are protected in agent sessions (classifier-judged edits); batch changes to them and call them out.
- The dependency stack and the pinned spack and spack-packages versions live in `.devcontainer/Dockerfile` and `.devcontainer/spack.yaml`; `docs/ONBOARDING.md` §2 describes the container.
