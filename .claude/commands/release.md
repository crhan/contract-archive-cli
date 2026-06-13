---
description: 发版到 PyPI（测试→构建→发布→验证 PyPI 200→打 tag→推送）
---

执行本项目发版，跑 `scripts/release.sh`。

**核心纪律：发布成功与否以 PyPI 的实际 HTTP 响应为准，绝不相信 `uv publish` 的命令回显。**
本项目曾因误信回显「以为发了其实没发」（PyPI 实则 404）。脚本第 7 步会轮询
`https://pypi.org/pypi/<pkg>/<version>/json` 直到 200 才算成功，没到 200 就不打 tag、不推送。

步骤：

1. 确认要发的版本已提交干净：`pyproject.toml` 的 `version` 已 bump、`CHANGELOG.md` 已定版、
   `git status` 干净。若还有未提交改动，先帮用户提交（**精准 `git add <file>`，禁止 `git add -A/.`**），
   再发版——脚本要求干净工作树且不会自动 commit。
2. 确认 token 就位：`.env` 里有 `UV_PUBLISH_TOKEN`（`.env` 已被 .gitignore 忽略，不入库）。
   **缺就停下找用户要，不要猜、不要从别处翻密钥。**
3. 跑 `scripts/release.sh`（先演练可用 `DRY_RUN=1 scripts/release.sh`，只测试+构建+校验不真发）。
4. 报告结果时，「发布成功」的依据必须是脚本里 PyPI 200 那步通过，而不是 publish 的输出。

版本号一旦发布不可覆盖；重发必须先 bump version。
