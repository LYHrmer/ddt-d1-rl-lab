# Pinned nlohmann JSON headers for native contract tests

These 41 modular headers are byte-identical copies of the installed Ubuntu
`nlohmann-json3-dev` package version `3.10.5-2` (upstream JSON for Modern C++
version `3.10.5`) from `/usr/include/nlohmann`. They total 804,087 bytes. All
original copyright and license notices remain intact. `COPYRIGHT` is an
unchanged copy of `/usr/share/doc/nlohmann-json3-dev/copyright`.

The frozen native tests already compile with the repository root in their
`-I` search path. Keeping this directory at `nlohmann/` allows their existing
`#include <nlohmann/json.hpp>` and its modular includes to resolve on CI
runners without a system installation. No frozen test, workflow, or runtime
source was changed to package this dependency.

Every copied header matches the SHA-256 recorded in both original runtime83
build receipts: `build_runtime83_01/build_receipt.json` and
`build_runtime83_checked_01/build_receipt.json`. The per-file system and copy
SHA-256 values, receipt hashes, package metadata checks, and license-copy hash
are recorded in `docs/experiments83/native_headers83_packaging.json`.

This is a new packaging input for future builds. The recorded experiments
used the system paths; their original binaries, dependency receipts, and
qualification evidence are unchanged. Packaging does not rerun an experiment,
replace its sealed sources, or establish hardware or timing qualification.
The next compilation must validate the repository include path separately.

Provenance is the local installed package and its dpkg metadata, plus the two
saved build dependency manifests. No package archive signature or separate
upstream release archive was fetched or independently verified in this step.
The full Debian copyright file also describes source-package components that
are not included in this header-only copy.

Upstream project: https://github.com/nlohmann/json
