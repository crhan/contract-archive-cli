#!/usr/bin/env bash
# 发版流程，把「别信命令回显、用 PyPI 实际响应验证」这条血泪教训焊进去。
# （本项目曾「以为发了其实没发」：publish 回显被误读成功，PyPI 实则 404。）
#
# 用法：先把要发的版本提交干净（version 已 bump、CHANGELOG 已定版），再跑本脚本。
#   scripts/release.sh            # 正式发布
#   DRY_RUN=1 scripts/release.sh  # 只测试+构建+校验，不 publish/tag/push
#
# token：从项目根 .env 读 UV_PUBLISH_TOKEN（.env 已被 .gitignore 忽略），或外部 env 注入。
set -euo pipefail
cd "$(dirname "$0")/.."

log() { printf '\033[1;34m[release]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[release] 失败:\033[0m %s\n' "$*" >&2; exit 1; }

PKG=$(grep -m1 '^name = ' pyproject.toml | sed -E 's/^name = "(.*)"/\1/')
VER=$(grep -m1 '^version = ' pyproject.toml | sed -E 's/^version = "(.*)"/\1/')
[ -n "$PKG" ] && [ -n "$VER" ] || die "无法从 pyproject.toml 解析 name/version"
TAG="v$VER"
log "包=$PKG  版本=$VER  tag=$TAG"

# 1) 工作树必须干净。脚本只发布「已提交、已审阅」的内容，绝不替你 commit——
#    自动提交会把未审阅的改动悄悄卷进发布，这正是要避免的。
[ -z "$(git status --porcelain)" ] || die "工作树有未提交改动；先 commit（精准 add）再发版"

# 2) 防重复发布：PyPI 版本号一旦用过不可覆盖，已存在就停。
if curl -fsS -o /dev/null "https://pypi.org/pypi/$PKG/$VER/json" 2>/dev/null; then
  die "$PKG $VER 已在 PyPI（版本号不可重用）；先 bump version"
fi

# 3) 测试必须绿，否则不发。
log "跑测试…"
uv run --extra dev --quiet pytest -q || die "测试未通过，中止发版"

# 4) 干净构建 + 产物校验。
log "构建 + twine check…"
rm -rf dist/
uv build >/dev/null || die "uv build 失败"
uvx twine check dist/* || die "twine check 失败"
ls -1 dist/

if [ "${DRY_RUN:-0}" = "1" ]; then
  log "DRY_RUN：跳过 publish/tag/push，产物在 dist/。"
  exit 0
fi

# 5) 凭证就位才发。
set -a; [ -f .env ] && . ./.env; set +a
[ -n "${UV_PUBLISH_TOKEN:-}" ] || die "缺 UV_PUBLISH_TOKEN（放 .env 或 export），无法 publish"

# 6) 发布。
log "发布到 PyPI…"
uv publish || die "uv publish 报错"

# 7) 核心纪律：不信 publish 回显，轮询 PyPI 直到该版本真的可见（HTTP 200）。
log "校验 PyPI 已索引 $VER（最多 ~2 分钟）…"
ok=0
for _ in $(seq 1 20); do
  if curl -fsS -o /dev/null "https://pypi.org/pypi/$PKG/$VER/json" 2>/dev/null; then ok=1; break; fi
  sleep 6
done
[ "$ok" = 1 ] || die "publish 后 PyPI 仍查不到 $VER —— 视作未成功，不打 tag/不推送"
log "PyPI 确认 $VER 已上线 ✓"

# 8) 只有 PyPI 确认后才打 tag + 推送（杜绝「tag 说发了、PyPI 其实没有」）。
git tag "$TAG"
git push origin HEAD
git push origin "$TAG"

# 9) 校验远端 tag 到位。
git ls-remote --tags origin "$TAG" | grep -q "refs/tags/$TAG" || die "tag $TAG 未推到远端，手动检查"
log "完成：$PKG $VER 已发布、$TAG 已推送 ✓"
