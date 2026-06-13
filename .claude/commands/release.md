---
description: 发版到 PyPI（推 tag 触发 GitHub Action 经 Trusted Publishing 发布，验证 PyPI 200）
---

执行本项目发版，跑 `scripts/release.sh`。

**发版机制：GitHub Action（`.github/workflows/release.yml`）经 PyPI Trusted Publishing 发布。**
推送 `v<version>` tag 触发 CI：测试 → 构建 → OIDC 可信发布。本地、仓库、secret 里都不存 token。

**核心纪律：发布成功与否以 PyPI 的实际 HTTP 200 为准，绝不相信任何命令/CI 回显。**
本项目曾因误信回显「以为发了其实没发」（PyPI 实则 404）。`scripts/release.sh` 第 5 步会轮询
`https://pypi.org/pypi/<pkg>/<version>/json` 直到 200 才算成功，没到就报错并提示去查 CI 日志。

步骤：

1. 确认要发的版本已提交干净：`pyproject.toml` 的 `version` 已 bump、`CHANGELOG.md` 已定版、
   `uv.lock` 已同步、`git status` 干净。若有未提交改动，先帮用户提交
   （**精准 `git add <file>`，禁止 `git add -A/.`**）——脚本要求干净工作树且不会自动 commit。
2. 跑 `scripts/release.sh`（先演练可用 `DRY_RUN=1 scripts/release.sh`，只测试+构建+校验、不推 tag）。
   它会：干净树检查 → 防重复发布 → 测试 → 构建+twine check → 推 main+tag → 轮询 PyPI 至 200。
3. 若 PyPI 迟迟不到 200：`gh run list` / `gh run view --log-failed` 看 CI 失败原因，别当成功。
4. 报告时，「发布成功」的依据必须是 PyPI 200，而不是 publish/CI 的输出。

前置（一次性，已配好则跳过）：PyPI 项目需配置 Trusted Publisher，绑定
owner=`crhan`、repo=`contract-archive-cli`、workflow=`release.yml`。版本号一旦发布不可覆盖，重发先 bump。
