#!/usr/bin/env bash
# clone-upstream.sh — shallow-clone the pinned audio.cpp commit into a target
# directory. Used by build-libs.yml to fetch 0xShug0/audio.cpp at the commit
# pinned in lib/version.txt before the shared lib is configured/built.
#
# The root CMakeLists.txt add_subdirectory()s this checkout (AUDIOCPP_SRC), so
# the dest must be the path AUDIOCPP_SRC points at (default: ./upstream/audio.cpp).
#
# Usage:
#   scripts/clone-upstream.sh <repo-url> <commit-ish> <dest-dir>
# e.g.
#   scripts/clone-upstream.sh https://github.com/0xShug0/audio.cpp.git \
#     "$(tr -d '[:space:]' < lib/version.txt)" upstream/audio.cpp
set -euo pipefail

url="${1:?usage: clone-upstream.sh <repo-url> <commit> <dest>}"
commit="${2:?missing commit}"
dest="${3:?missing dest}"

mkdir -p "$dest"
cd "$dest"
git init -q
git remote add origin "$url" 2>/dev/null || git remote set-url origin "$url"

# A specific SHA can't always be fetched shallowly (server may refuse
# uploadpack.allowReachableSHA1InWant); fall back to a full fetch + checkout.
if git fetch --depth 1 origin "$commit" 2>/dev/null; then
  git checkout -q FETCH_HEAD
else
  echo "clone-upstream: direct shallow fetch of $commit refused; fetching history" >&2
  git fetch origin
  git checkout -q "$commit"
fi

# audio.cpp vendors ggml (and others) as submodules; the shim links them
# statically, so they must be present at configure time.
git submodule update --init --recursive --depth 1
echo "clone-upstream: $url @ $(git rev-parse --short HEAD) checked out into $dest"
