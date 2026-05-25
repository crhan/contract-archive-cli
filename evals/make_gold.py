"""
从已入库的真实合同生成**脱敏 draft gold** 评测 case（模型辅助标注，省人工）。

数据流：archive 的 mineru 产物（raw_text/markdown，复用生产 _load_document_text 保证与
生产喂给 extract_document 的文本一致）+ extraction_result.json（champion 抽取，当 draft gold）
→ 脱敏（PII 占位）→ 落 gitignored 的 evals/cases_private/extraction/<id>/。

**安全红线**：
- 输出**只**写 cases_private/（已 gitignore），绝不写可提交的 cases/。
- archive 里是真实 PII（真人名/真公司/身份证/电话）。本工具按抽取出的实体 + 正则兜底脱敏，
  但**可能漏**（正文里未被抽取的人名等）。所以产物是 DRAFT：人工读一遍 input.txt 确认无残留
  真实 PII、并**盲标**高风险字段（parties/amounts/issues 对照原文核），确认后才手动提升到 cases/。
- draft gold 来自 champion 单源 → 有 champion 盲区偏向。建议带 --crosscheck <异家族模型>，
  工具会在脱敏文本上再跑一个不同家族模型，产出 crosscheck.json 供人工对比、补 champion 漏抽。

用法：
  uv run --no-sync python -m evals.make_gold                       # 处理 archive 全部已入库文档
  uv run --no-sync python -m evals.make_gold --doc-id e9d1809860f7 # 只处理某文档
  uv run --no-sync python -m evals.make_gold --crosscheck deepseek-v4-pro
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from contract_archive.archive.ingest import _load_document_text
from contract_archive.config import load_settings
from contract_archive.extraction.llm_extractor import _call_openai_compat, _parse_json_loose
from contract_archive.schemas import DocumentExtraction

logger = logging.getLogger(__name__)

DEID_PROMPT = """你是严谨的脱敏助手。下面是一份文档的文本，找出其中所有能识别到具体个人或机构的
隐私信息(PII)，给出替换映射。只输出 JSON，不要任何解释或 markdown。

需脱敏（给占位符，同一实体全文用同一占位）：
- 人名 → 张三、李四、王五、赵六、孙七…（按出现顺序）
- 公司/机构/品牌名（**中文与英文都要**，如 EXAMPLE GROUP、示例集团）→ 示例置业有限公司、示例科技有限公司…
- 身份证号 → 3301xxxxxxxxxxxxxx
- 手机号 → 138xxxxxxxx
- 座机/传真/分机号（含区号与分机）→ 0000-0000000
- 电子邮箱 → user@example.com
- 详细地址 → 示例市示例区示例路1号
- 邮政编码 → 000000
- 银行账号/卡号 → 6222xxxxxxxxxxxx

不要脱敏：金额、日期、通用职务词（HR/工程师/经理）、条款编号、通用名词。

输出（只列需替换项，original 必须是文中**原样**出现的子串，按 original 长度降序）：
{"replacements":[{"original":"原文串","placeholder":"占位串"}]}
"""

EVALS_DIR = Path(__file__).resolve().parent
CASES_PRIVATE = EVALS_DIR / "cases_private" / "extraction"

# 占位池（与种子 case 同风格，便于一致）
PERSON_POOL = ["张三", "李四", "王五", "赵六", "孙七", "周八", "吴九", "郑十", "冯十一", "陈十二"]
COMPANY_POOL = [
    "示例置业有限公司", "示例科技有限公司", "示例商贸有限公司", "示例物业管理有限公司",
    "示例实业有限公司", "示例银行股份有限公司", "示例咨询有限公司", "示例建设有限公司",
]
COMPANY_HINTS = ("公司", "有限", "集团", "银行", "事务所", "中心", "厂", "店", "医院", "学校",
                 "大学", "政府", "委员会", "管理局", "合作社", "协会", "基金", "证券", "保险",
                 "实业", "置业", "物业", "商贸", "科技", "技术", "控股", "传媒", "网络")
NAME_LABEL_HINTS = ("持证人", "姓名", "法定代表人", "代表人", "联系人", "经办人", "委托代理人",
                    "受让方", "转让方", "买受人", "出卖人", "承租方", "出租方", "甲方", "乙方",
                    "丙方", "购买方", "销售方", "开票方", "收款人", "付款人", "员工", "用人单位",
                    "单位", "公司名称")
ID_LABEL = ("身份证", "证件号", "纳税人识别号", "统一社会信用代码", "信用代码")
PHONE_LABEL = ("电话", "手机", "联系方式", "总机", "传真", "座机", "分机")
ADDR_LABEL = ("地址", "住所", "住址")
POSTAL_LABEL = ("邮编", "邮政编码")

ID_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")          # 18 位身份证
PHONE_RE = re.compile(r"(?<!\d)1\d{10}(?!\d)")             # 11 位手机
LANDLINE_RE = re.compile(r"(?<!\d)\d{3,4}-\d{7,8}(?!\d)")  # 座机 区号-号码（0571-xxxxxxxx）
ID_MASK, PHONE_MASK, LANDLINE_MASK = "3301xxxxxxxxxxxxxx", "138xxxxxxxx", "0000-0000000"
# 切分自由文本以捞嵌入的机构名（如"示例集团-营销平台..."里的"示例集团"）
_SEG_SPLIT = re.compile(r"[-—－/、,，()（）\s]+")


def _is_company(s: str) -> bool:
    return any(h in s for h in COMPANY_HINTS)


def _company_segments(s: str) -> list[str]:
    """从自由文本切出含公司线索的片段（≥3 字），捞未被抽取的嵌入机构名。"""
    return [seg for seg in _SEG_SPLIT.split(s or "") if len(seg) >= 3 and _is_company(seg)]


def _uniq(items: list[str]) -> list[str]:
    """去重保序，去空白。"""
    out: list[str] = []
    for x in items:
        x = (x or "").strip()
        if x and x not in out:
            out.append(x)
    return out


def build_deid_map(env: DocumentExtraction) -> dict[str, str]:
    """从抽取出的实体建脱敏映射 {真实串: 占位串}。dates/amounts 不动（非 PII 且 gold 需要）。"""
    persons: list[str] = []
    companies: list[str] = []
    addresses: list[str] = []
    ids: list[str] = []
    phones: list[str] = []
    postals: list[str] = []

    def add_entity(s: str) -> None:
        (companies if _is_company(s) else persons).append(s)

    # 自由文本里嵌入的机构名（职位/摘要等）也要捞出来脱敏
    free_texts: list[str] = [env.title or "", env.summary or ""]
    for p in env.parties:
        add_entity(p)
    for s in env.seals:
        if s.owner:
            add_entity(s.owner)
    for f in env.fields:
        label, val = f.label or "", (f.value or "").strip()
        if not val:
            continue
        if any(h in label for h in ID_LABEL):
            ids.append(val)
        elif any(h in label for h in PHONE_LABEL):
            phones.append(val)
        elif any(h in label for h in POSTAL_LABEL):
            postals.append(val)
        elif any(h in label for h in ADDR_LABEL):
            addresses.append(val)
        elif any(h in label for h in NAME_LABEL_HINTS):
            add_entity(val)
        else:
            free_texts.append(val)   # 其他字段值（如职位）可能嵌机构名
    for t in free_texts:
        companies.extend(_company_segments(t))

    mapping: dict[str, str] = {}

    def assign(reals: list[str], pool: list[str], fallback: str) -> None:
        for i, r in enumerate(_uniq(reals)):
            mapping[r] = pool[i] if i < len(pool) else f"{fallback}{i + 1}"

    assign(persons, PERSON_POOL, "示例人")
    assign(companies, COMPANY_POOL, "示例机构")
    for i, a in enumerate(_uniq(addresses)):
        mapping[a] = f"示例市示例区示例路{i + 1}号"
    for x in _uniq(ids):
        mapping[x] = ID_MASK
    for x in _uniq(phones):
        mapping[x] = PHONE_MASK
    for x in _uniq(postals):
        mapping[x] = "000000"
    return mapping


def llm_build_deid_map(text: str, model: str) -> dict[str, str]:
    """
    LLM 脱敏：识别上下文里的人名/中英文机构名/各类号码，返回 {原文串: 占位串}。
    规则法兜不住自由文本里的人名与英文机构名，故以 LLM 为主、正则为兜底。失败返回 {}。
    """
    settings = load_settings()
    if not settings.dashscope_api_key:
        logger.warning("无 API key，跳过 LLM 脱敏（仅用规则，残留风险更高）")
        return {}
    try:
        content, _ = _call_openai_compat(
            DEID_PROMPT, f"文本如下：\n\n{text}", model,
            settings.dashscope_api_key, settings.dashscope_base_url)
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM 脱敏调用失败，仅用规则: %s", e)
        return {}
    out: dict[str, str] = {}
    for r in _parse_json_loose(content).get("replacements", []):
        if not isinstance(r, dict):
            continue
        o, p = str(r.get("original", "")).strip(), str(r.get("placeholder", "")).strip()
        if o and p and o != p:
            out[o] = p
    return out


def deidentify_text(text: str, mapping: dict[str, str]) -> str:
    """先按映射长键优先替换（避免子串误伤），再用正则兜底掉漏网的身份证/手机。"""
    for k in sorted(mapping, key=len, reverse=True):
        text = text.replace(k, mapping[k])
    text = ID_RE.sub(ID_MASK, text)
    text = PHONE_RE.sub(PHONE_MASK, text)
    text = LANDLINE_RE.sub(LANDLINE_MASK, text)
    return text


def deidentify_json(obj: Any, mapping: dict[str, str]) -> Any:
    """递归对 JSON 里所有字符串做脱敏。"""
    if isinstance(obj, str):
        return deidentify_text(obj, mapping)
    if isinstance(obj, list):
        return [deidentify_json(x, mapping) for x in obj]
    if isinstance(obj, dict):
        return {k: deidentify_json(v, mapping) for k, v in obj.items()}
    return obj


_DIGIT_RUN = re.compile(r"\d{7,}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_UPPER_EN = re.compile(r"\b[A-Z][A-Za-z&. ]{3,}\b")


def residual_flags(text: str) -> list[str]:
    """启发式扫脱敏后仍可疑的 token（辅助人工复核，非保证）。占位符自身(全0/全x)不算。"""
    flags: set[str] = set()
    for m in _DIGIT_RUN.findall(text):
        if set(m) != {"0"}:          # 排除 0000000 这类掩码
            flags.add(m)
    flags.update(_EMAIL.findall(text))
    for m in _UPPER_EN.findall(text):
        if m.strip() not in ("HR",):
            flags.add(m.strip())
    return sorted(flags)


def iter_archive_docs(archive_dir: Path, only: Optional[str]) -> list[tuple[str, Path, Path]]:
    """列出 archive 里有 mineru/ + extraction_result.json 的文档 → (doc_id, mineru_dir, result_json)。"""
    docs_dir = archive_dir / "documents"
    out: list[tuple[str, Path, Path]] = []
    if not docs_dir.is_dir():
        return out
    for d in sorted(p for p in docs_dir.iterdir() if p.is_dir()):
        if only and d.name != only:
            continue
        mineru, result = d / "mineru", d / "extraction_result.json"
        if mineru.is_dir() and result.exists():
            out.append((d.name, mineru, result))
    return out


REVIEW_TEMPLATE = """# DRAFT case 人工核对清单（{doc_id}）

⚠️ 本 case 由 make_gold 从真实合同自动生成并脱敏，**未经人工核对，不可直接提升到 cases/**。

## 1. 脱敏完整性（PII 红线，最高优先级）
- [ ] 通读 input.txt，确认**无任何残留真实 PII**（人名/公司/身份证/电话/地址/账号）。
      工具按抽取实体+正则脱敏，正文里未被抽取的人名等**可能漏网**——逐行扫一遍。
- [ ] gold.json 同样无残留真实 PII。

## 2. gold 正确性（破除 champion 盲区，盲标高风险字段）
- [ ] **不看模型输出**，对照 input.txt 原文，盲标：parties / amounts(数值+is_total_component)
      / completeness.issues（缺什么、页码出处）。再与 gold.json diff，以你的盲标为准修正。
- [ ] 若有 crosscheck.*.json：对比它与 gold，凡 champion 漏抽而 crosscheck 抽到的，回原文核实补上。
- [ ] doc_type / 日期(ISO) / seals / sub_agreements 核一遍。

## 3. 提升为正式 case
确认 1、2 后：把 input.txt + 修正后的 gold.json + meta.json 复制到
`evals/cases/extraction/<有意义的id>/`，git add 提交。meta.json 记好 stratum/difficulty。
"""


def write_case(doc_id: str, text: str, env_json: dict, crosscheck: Optional[dict]) -> Path:
    case_dir = CASES_PRIVATE / doc_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "input.txt").write_text(text, encoding="utf-8")
    (case_dir / "gold.json").write_text(json.dumps(env_json, ensure_ascii=False, indent=2), encoding="utf-8")
    (case_dir / "meta.json").write_text(json.dumps({
        "doc_type": env_json.get("doc_type"),
        "stratum": "real-derived(待人工归类)",
        "difficulty": "unknown",
        "provenance": "make_gold 从真实合同生成并脱敏的 DRAFT；未经人工核对，禁止入库",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (case_dir / "REVIEW.md").write_text(REVIEW_TEMPLATE.format(doc_id=doc_id), encoding="utf-8")
    if crosscheck is not None:
        (case_dir / "crosscheck.json").write_text(
            json.dumps(crosscheck, ensure_ascii=False, indent=2), encoding="utf-8")
    return case_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="从真实合同生成脱敏 draft gold（落 cases_private/）")
    ap.add_argument("--archive-dir", type=Path, default=None, help="默认读 config 的 archive 目录")
    ap.add_argument("--doc-id", default=None, help="只处理某个文档（archive documents/ 下的 id）")
    ap.add_argument("--crosscheck", default=None,
                    help="用一个异家族模型在脱敏文本上再抽一遍，产 crosscheck.json 供人工对比（破 champion 盲区）")
    ap.add_argument("--deid-model", default=None, help="LLM 脱敏所用模型（默认 settings 的文本模型）")
    ap.add_argument("--no-llm-deid", action="store_true",
                    help="只用规则脱敏，不调 LLM（残留风险更高，不建议）")
    args = ap.parse_args(argv)

    archive_dir = args.archive_dir
    if archive_dir is None:
        settings = load_settings()
        archive_dir = Path(settings.archive_dir) if settings.archive_dir else \
            Path.home() / ".local/share/contract-archive"
    docs = iter_archive_docs(archive_dir, args.doc_id)
    if not docs:
        print(f"⚠️  {archive_dir}/documents 下没有可用文档（需含 mineru/ + extraction_result.json）")
        return 1

    print(f"archive: {archive_dir}　待处理 {len(docs)} 个文档　→ 输出到 {CASES_PRIVATE}（gitignore）\n")
    for doc_id, mineru_dir, result_path in docs:
        raw_text = _load_document_text(mineru_dir)
        if not raw_text.strip():
            print(f"  跳过 {doc_id}：mineru 文本为空")
            continue
        env = DocumentExtraction.model_validate(json.loads(result_path.read_text(encoding="utf-8")))
        mapping = build_deid_map(env)   # 规则：结构化实体 + 号码
        if not args.no_llm_deid:        # LLM 主力：自由文本里的人名/中英文机构名（覆盖规则）
            mapping = {**mapping, **llm_build_deid_map(
                raw_text, args.deid_model or load_settings().dashscope_model)}
        deid_text = deidentify_text(raw_text, mapping)
        deid_gold = deidentify_json(json.loads(result_path.read_text(encoding="utf-8")), mapping)
        deid_gold["llm_model"] = None  # gold 不带抽取来源
        deid_gold["llm_usage"] = None

        crosscheck = None
        if args.crosscheck:
            from contract_archive.extraction import extract_document
            cc = extract_document(deid_text, model=args.crosscheck)
            crosscheck = cc.model_dump(mode="json")

        case_dir = write_case(doc_id, deid_text, deid_gold, crosscheck)
        flags = residual_flags(deid_text + "\n" + json.dumps(deid_gold, ensure_ascii=False))
        if flags:
            (case_dir / "SCAN.txt").write_text(
                "脱敏后仍可疑的 token（人工逐个确认是否需脱敏）：\n" + "\n".join(flags),
                encoding="utf-8")
        warn = f"  ⚠️ 残留可疑 {len(flags)} 处(见 SCAN.txt)" if flags else "  扫描无明显残留"
        print(f"  ✓ {doc_id}: doc_type={deid_gold.get('doc_type')} 脱敏实体 {len(mapping)} 个"
              f" → {case_dir.relative_to(EVALS_DIR.parent)}{warn}")

    print("\n⚠️  这些是 DRAFT：先按各 case 的 REVIEW.md 人工核对脱敏完整性 + 盲标高风险字段，"
          "确认后再手动提升到 evals/cases/extraction/ 提交。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
