"""
主体身份基准库（known_parties）：跨文档核对主体的固有标识。

为什么独立于 db.sqlite：这是"基本信息基准"而非"文档档案"——它跨文档累积
（某主体的身份证号一次录入、之后每份文档都拿来核对），生命周期与单份文档解耦。
存档案库根目录 known_parties.json，含真实 PII，故文件权限 0600、列入 .gitignore。

核对模型（用户要的"首见入库、再见校对"）：
  - 某主体的某标识首次出现 → 录入为基准，记首见出处。
  - 之后同主体同标识再出现 → 与基准比对，不一致即报 identity 缺陷（不覆盖基准，
    基准保持稳定；要修正基准用 `party set`）。
  - 不分自然人/机构：身份证、电话、银行账号、开户行、税号一律核对。

归一化：比较时去除空白与常见分隔符（OCR 把"；"读成"："、夹空格等不算差异），
但保留真实数字差异（多一位/少一位/改一位）——后者正是要抓的 OCR 读错/篡改。
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..schemas import CompletenessIssue, PersonIdentity

logger = logging.getLogger(__name__)

REGISTRY_VERSION = 1

# 比较前剥离的噪声字符：空白 + 常见分隔/标点。不动数字、字母、汉字本身。
_NOISE_RE = re.compile(r"[\s;；,，、:：.。\-—_／/]")


def _canon(value: str) -> str:
    """归一化用于比较：去首尾空白 + 剥离分隔/标点噪声。

    使 OCR 分隔符差异（空格、"；"读成"："等）不误报；但多一位/少一位/改一位
    这类真实数字差异会保留下来——那正是要抓的 OCR 读错或信息被改。
    """
    return _NOISE_RE.sub("", value.strip())


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class PartyRegistry:
    """known_parties.json 的读写 + 首见入库/再见校对。"""

    def __init__(self, path: Path, data: Optional[dict] = None) -> None:
        self._path = path
        self._data = data if data is not None else {"version": REGISTRY_VERSION, "parties": {}}
        self._dirty = False

    # ---------- 加载 / 保存 ----------

    @classmethod
    def load(cls, path: Path) -> "PartyRegistry":
        """读基准库；文件不存在/损坏/结构非法一律返回空库——只读路径必须健壮，坏文件不能让入库崩。"""
        if not path.exists():
            return cls(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("known_parties 读取失败，按空库处理: %s", e)
            return cls(path)
        if not isinstance(payload, dict) or not isinstance(payload.get("parties"), dict):
            logger.warning("known_parties 结构非法，按空库处理: %s", path)
            return cls(path)
        return cls(path, payload)

    def save(self) -> Path:
        """写基准库；文件 0600（含 PII，仅本人可读，每次都 chmod 防 umask 宽松）。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        self._path.chmod(0o600)
        self._dirty = False
        return self._path

    @property
    def dirty(self) -> bool:
        """自上次 load/save 以来是否有录入改动（调用方据此决定要不要 save）。"""
        return self._dirty

    # ---------- 核对（首见入库 / 再见校对）----------

    def reconcile(self, identities: list[PersonIdentity], doc_sha: str) -> list[CompletenessIssue]:
        """
        把一份文档抽到的 person_identities 与基准库比对。

        首见（基准无此 主体·标识）→ 录入基准并记首见出处；
        再见（基准已有）→ 比对，不一致返回 identity 缺陷（不覆盖基准）。
        就地修改基准库（录入首见），是否落盘由调用方 save 决定（看 dirty）。
        """
        issues: list[CompletenessIssue] = []
        parties = self._data["parties"]
        for person in identities:
            name = person.name.strip()
            if not name:
                continue
            slot = parties.setdefault(name, {})
            for idv in person.identifiers:
                label, value = idv.label.strip(), idv.value.strip()
                if not label or not value:
                    continue
                known = slot.get(label)
                if known is None:
                    slot[label] = {
                        "value": value,
                        "first_seen_doc": doc_sha,
                        "first_seen_at": _now_iso(),
                        "role": person.role or "",
                    }
                    self._dirty = True
                elif _canon(known.get("value", "")) != _canon(value):
                    base = known.get("value", "")
                    src = str(known.get("first_seen_doc", ""))[:12]
                    issues.append(CompletenessIssue(
                        item=f"{name}·{label}",
                        category="identity",
                        detail=(
                            f"与基准不一致：基准『{base}』(首见于 {src})，"
                            f"本次『{value}』——疑似 OCR 读错或信息被改，请人工核对"
                        ),
                        evidence=f"本次文档 {doc_sha[:12]}",
                    ))
        return issues

    # ---------- 管理（party 命令组用）----------

    def all_parties(self) -> dict:
        """全部基准：name → {label → {value, first_seen_doc, first_seen_at, role}}。"""
        return self._data["parties"]

    def get(self, name: str) -> Optional[dict]:
        """某主体的全部标识基准；无则 None。"""
        return self._data["parties"].get(name.strip())

    def set(self, name: str, label: str, value: str) -> None:
        """手动录入/修正基准（覆盖既有值，来源标记为 manual，便于 show 区分）。"""
        name, label, value = name.strip(), label.strip(), value.strip()
        if not name or not label or not value:
            raise ValueError("name/label/value 均不能为空")
        slot = self._data["parties"].setdefault(name, {})
        slot[label] = {
            "value": value,
            "first_seen_doc": "(manual)",
            "first_seen_at": _now_iso(),
            "role": slot.get(label, {}).get("role", ""),
        }
        self._dirty = True

    def remove(self, name: str, label: Optional[str] = None) -> bool:
        """删某主体的某标识；label 省略则删整个主体。返回是否真的删到。"""
        name = name.strip()
        parties = self._data["parties"]
        if name not in parties:
            return False
        if label is None:
            del parties[name]
            self._dirty = True
            return True
        label = label.strip()
        if label in parties[name]:
            del parties[name][label]
            if not parties[name]:        # 主体下已无任何标识，清掉空壳
                del parties[name]
            self._dirty = True
            return True
        return False
