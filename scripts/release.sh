#!/usr/bin/env bash
# 发版：推 git tag 触发 GitHub Action（.github/workflows/release.yml）经 PyPI
# Trusted Publishing 发布。把「别信回显、用 PyPI 实际 200 验证」焊进流程——
# 本项目曾「以为发了其实没发」（publish 回显被误读成功，PyPI 实则 404）。
#
# 用法：先把要发的版本提交干净（version 已 bump、CHANGELOG 已定版），再跑本脚本。
#   scripts/release.sh            # 正式发布
#   DRY_RUN=1 scripts/release.sh  # 只测试+构建+校验，不推 tag、不触发 CI
#
# 凭证：本地无需任何 token。发布由 CI 经 Trusted Publishing 完成（OIDC，无长期密钥）。
set -euo pipefail
cd "$(dirname "$0")/.."

log() { printf '\033[1;34m[release]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[release] 失败:\033[0m %s\n' "$*" >&2; exit 1; }

PKG=$(grep -m1 '^name = ' pyproject.toml | sed -E 's/^name = "(.*)"/\1/')
VER=$(grep -m1 '^version = ' pyproject.toml | sed -E 's/^version = "(.*)"/\1/')
[ -n "$PKG" ] && [ -n "$VER" ] || die "无法从 pyproject.toml 解析 name/version"
TAG="v$VER"
log "包=$PKG  版本=$VER  tag=$TAG"

# 1) 工作树必须干净。只发布「已提交、已审阅」的内容，绝不替你 commit。
[ -z "$(git status --porcelain)" ] || die "工作树有未提交改动；先 commit（精准 add）再发版"

# 2) 防重复发布：PyPI 版本号一旦用过不可覆盖，已存在就停。
if curl -fsS -o /dev/null "https://pypi.org/pypi/$PKG/$VER/json" 2>/dev/null; then
  die "$PKG $VER 已在 PyPI（版本号不可重用）；先 bump version"
fi

# 3) 本地最后一道闸：测试 + 构建 + 产物校验（CI 里还会再跑一遍 test）。
log "本地测试…"; uv run --extra dev --quiet pytest -q || die "测试未通过，中止发版"
log "本地构建 + twine check…"; rm -rf dist/; uv build >/dev/null || die "uv build 失败"
uvx twine check dist/* || die "twine check 失败"

if [ "${DRY_RUN:-0}" = "1" ]; then
  log "DRY_RUN：跳过 push/tag，不触发 CI。产物在 dist/。"; exit 0
fi

# 4) 推 main + 打 tag 推 tag → 触发 CI 经 Trusted Publishing 发布。
log "推送 main…"; git push origin HEAD
log "打 tag 并推送 → 触发 CI 发布…"; git tag "$TAG"; git push origin "$TAG"

# 5) 核心纪律：不信任何回显，轮询 PyPI 直到该版本真的可见（HTTP 200）。
#    等的是 CI 跑完（spin-up + 测试 + 构建 + 上传），放宽到最多 ~6 分钟。
log "等 CI 发布，并校验 PyPI 出现 $VER（最多 ~6 分钟）…"
ok=0
for _ in $(seq 1 36); do
  if curl -fsS -o /dev/null "https://pypi.org/pypi/$PKG/$VER/json" 2>/dev/null; then ok=1; break; fi
  sleep 10
done
[ "$ok" = 1 ] || die "等了 ~6 分钟 PyPI 仍无 $VER —— 别当成功；查 CI 日志：gh run list / gh run view --log-failed"
log "PyPI 确认 $VER 已上线 ✓（由 GitHub Action 经 Trusted Publishing 发布）"
