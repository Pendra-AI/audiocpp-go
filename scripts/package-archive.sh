#!/usr/bin/env bash
# package-archive.sh — stage a built libaudiocpp plus the licences of everything
# statically linked into it, and tar it up. Used by build-libs.yml (Unix legs).
#
# The shared lib bundles audio.cpp + ggml + sentencepiece (all hidden), so their
# licences must travel with the redistributed binary.
#
# Usage: scripts/package-archive.sh <lib-path> <archive-name> <upstream-dir>
set -euo pipefail

lib="${1:?usage: package-archive.sh <lib-path> <archive-name> <upstream-dir>}"
name="${2:?missing archive name}"
upstream="${3:?missing upstream dir}"

stage="stage"
rm -rf "$stage"; mkdir -p "$stage/licenses"
cp "$lib" "$stage/"

# audio.cpp's own licence.
for f in LICENSE LICENSE.txt COPYING NOTICE; do
  [ -f "$upstream/$f" ] && cp "$upstream/$f" "$stage/licenses/audio.cpp-$f"
done
# Vendored deps' licences, wherever they landed after submodule init (ggml,
# sentencepiece, and any others). Best-effort: copy the first LICENSE found per
# vendored tree so the archive carries attribution without guessing paths.
if [ -d "$upstream" ]; then
  while IFS= read -r lic; do
    # name the copy by its parent dir so multiple licences don't collide
    parent="$(basename "$(dirname "$lic")")"
    cp "$lic" "$stage/licenses/${parent}-$(basename "$lic")" 2>/dev/null || true
  done < <(find "$upstream" -maxdepth 4 -type f \
             \( -iname 'LICENSE' -o -iname 'LICENSE.*' -o -iname 'COPYING' \) \
             -path '*ggml*' -o -path '*sentencepiece*' 2>/dev/null | head -20)
fi

cat > "$stage/NOTICE" <<EOF
This archive contains libaudiocpp, a self-contained shared library that
statically links audio.cpp (0xShug0/audio.cpp) and its dependencies (ggml,
sentencepiece, and vendored cjson/yaml). Their licences are under licenses/.
The audiocpp-go binding itself is licensed per the repository LICENSE.
EOF

tar -C "$stage" -czf "$name" .
echo "Packaged $name"
tar -tzf "$name" | sed 's/^/  /'
