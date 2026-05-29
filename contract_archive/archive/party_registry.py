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

实体对齐（key 不是字面 name，而是"实体"）：同一实体在不同文档/同一文档内常被
识别成不同名字（LLM 幻觉改字、称谓差异、OCR 误读），若按字面 name 作 key 就会
分裂、跨文档核对不到一起。故归位规则：
  - 主体名先规范化（剥离"甲方：/出卖人："等分隔符门控的称谓前缀）。
  - 强标识（身份证/银行账号/印章/统一社会信用代码/税号）实体唯一，同值必同实体——
    本次某强标识值若已登记在另一 name 下，即归并到那个已有 key，并把本次 name 记入
    别名表，今后即使只带弱标识也能归位。
  - 弱标识（电话、开户行）多人共用（公司总机、银行支行），绝不据此合并，避免误并。
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

REGISTRY_VERSION = 2  # v2 起新增 aliases（实体归并别名表）；v1 文件仍可读，缺则按空表

# 比较前剥离的噪声字符：空白 + 常见分隔/标点。不动数字、字母、汉字本身。
_NOISE_RE = re.compile(r"[\s;；,，、:：.。\-—_／/]")

# 称谓前缀：name 开头的"甲方/乙方/出卖人…"+ 分隔符，规范化时剥离，使
# "甲方：示例置业"与"示例置业"归到同一 key。仅当前缀后紧跟分隔符才剥离，
# 避免误伤"甲方物流有限公司"这类前缀恰是名字一部分的合法名。
_ROLE_PREFIXES = (
    "甲方", "乙方", "丙方", "出卖人", "买受人", "出租方", "承租方",
    "转让方", "受让方", "委托方", "受托方", "持证人", "卖方", "买方",
)
# 前缀后须接"分隔标点"或"一段空白"才剥离；二者皆无（如"甲方物流"）则前缀属名字本身，不剥。
_ROLE_PREFIX_RE = re.compile(
    r"^(?:" + "|".join(_ROLE_PREFIXES) + r")(?:\s*[:：、|/\\\-]+\s*|\s+)"
)

# 强标识 label 关键字：这些标识实体唯一（同值必同实体），可据此把同实体的不同
# name 变体归并到一个 key。电话/开户行是弱标识（多人共用），故意不在此列。
_STRONG_LABEL_KEYS = ("身份证", "银行账", "印章", "信用代码", "税号")


def _canon(value: str) -> str:
    """归一化用于比较：去首尾空白 + 剥离分隔/标点噪声。

    使 OCR 分隔符差异（空格、"；"读成"："等）不误报；但多一位/少一位/改一位
    这类真实数字差异会保留下来——那正是要抓的 OCR 读错或信息被改。
    """
    return _NOISE_RE.sub("", value.strip())


def _canon_name(name: str) -> str:
    """主体名规范化作 registry key：去首尾空白 + 剥离分隔符门控的称谓前缀。"""
    return _ROLE_PREFIX_RE.sub("", name.strip())


def group_by_value(ids: dict) -> list[tuple[str, dict]]:
    """把同一主体下『归一化值相同』的多个 label 折叠成一组，供人读展示去冗余。

    同一个号被不同文档写成不同 label（如『电话』『联系电话』）时，在 party list/show
    里并排堆着是纯噪声、零信息。canon 值相等即视为同一事实，标签并成『电话/联系电话』，
    rec 取首个（即基准首见那条）。值不同的 label（如公司总机 vs 联系人线）各自独立、
    绝不合并——与 reconcile『弱标识不据此并实体』同一立场：这里只折叠展示、不动数据。

    判等用 reconcile 同一套 _canon，保证『展示折叠』与『一致性校对』口径一致：
    凡 reconcile 视作"无差异"的两值，这里才折叠；有真实数字差异的不会被并掉。
    保持各组首次出现的插入顺序。

    Args:
        ids: 某主体的标识基准，label -> rec（rec 含 value/first_seen_doc/role 等）。
    Returns:
        [(合并后标签, 首个rec), ...]，按各组首次出现顺序排列。
    """
    labels_of: dict[str, list[str]] = {}   # canon(value) -> 同值的 label 列表
    rep_rec: dict[str, dict] = {}          # canon(value) -> 首个 rec（基准首见那条）
    order: list[str] = []                  # canon(value) 首次出现顺序，定输出顺序
    for label, rec in ids.items():
        key = _canon(rec.get("value") or "")
        if key not in labels_of:
            labels_of[key] = []
            rep_rec[key] = rec
            order.append(key)
        labels_of[key].append(label)
    return [("/".join(labels_of[key]), rep_rec[key]) for key in order]


def _is_strong_label(label: str) -> bool:
    """该标识是否为实体唯一的强标识（可据此把不同 name 归并为同一实体）。"""
    return any(k in label for k in _STRONG_LABEL_KEYS)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class PartyRegistry:
    """known_parties.json 的读写 + 首见入库/再见校对。"""

    def __init__(self, path: Path, data: Optional[dict] = None) -> None:
        self._path = path
        self._data = data if data is not None else {
            "version": REGISTRY_VERSION, "parties": {}, "aliases": {},
        }
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
        把一份文档抽到的 person_identities 与基准库比对（按实体对齐，非字面 name）。

        每个主体先经 _resolve_entity 定位 canonical key（别名/强标识归并）：
        首见（canonical 无此标识）→ 录入基准并记首见出处；
        再见（canonical 已有）→ 比对，不一致返回 identity 缺陷（不覆盖基准）。
        就地修改基准库（录入首见 + 学到的别名），是否落盘由调用方 save 决定（看 dirty）。
        """
        issues: list[CompletenessIssue] = []
        parties = self._data["parties"]
        aliases = self._data.setdefault("aliases", {})   # name 变体 → canonical（v1 文件无此键）
        for person in identities:
            if not person.name.strip():
                continue
            name = _canon_name(person.name)
            canonical = self._resolve_entity(name, person, parties, aliases)
            if canonical != name and aliases.get(name) != canonical:
                aliases[name] = canonical            # 记下别名，今后只带弱标识也能归位
                self._dirty = True
            slot = parties.setdefault(canonical, {})
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
                        item=f"{canonical}·{label}",
                        category="identity",
                        detail=(
                            f"与基准不一致：基准『{base}』(首见于 {src})，"
                            f"本次『{value}』——疑似 OCR 读错或信息被改，请人工核对"
                        ),
                        evidence=f"本次文档 {doc_sha[:12]}",
                    ))
        return issues

    def _resolve_entity(
        self, name: str, person: PersonIdentity, parties: dict, aliases: dict
    ) -> str:
        """
        定位本主体应归入的 canonical key（实体对齐，而非字面 name）：
          1. 已是已知别名 → 直达其 canonical；
          2. 已是现有 key → 用自己；
          3. 本次某强标识值已登记在另一 name 下 → 同实体，归并到那个已有 key；
          4. 都不是 → 新实体，用规范化后的 name。

        只用强标识（身份证/银行账号/印章/信用代码/税号）归并——它们实体唯一，
        同值必同实体；弱标识（电话/开户行）多人共用，绝不据此合并。
        """
        if name in aliases:
            return aliases[name]
        if name in parties:
            return name
        for idv in person.identifiers:
            if not _is_strong_label(idv.label):
                continue
            value = _canon(idv.value.strip())
            if not value:
                continue
            for existing_name, slot in parties.items():
                for label, info in slot.items():
                    if _is_strong_label(label) and _canon(info.get("value", "")) == value:
                        return existing_name
        return name

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
