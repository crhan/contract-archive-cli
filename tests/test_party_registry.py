"""
PartyRegistry 首见入库 / 再见校对 行为测试。

PII 一律占位：张三 + 明显虚构的号码（110101199001011234 等），绝不用真实身份证。
"""
from contract_archive.archive.party_registry import (
    PartyRegistry,
    _canon,
    _canon_name,
    _is_strong_label,
    group_by_value,
)
from contract_archive.schemas import LabeledValue, PersonIdentity


def _person(name, role, **ids):
    """便捷构造 PersonIdentity：_person('张三', '乙方', 身份证号='...', 电话='...')。"""
    return PersonIdentity(
        name=name,
        role=role,
        identifiers=[LabeledValue(label=k, value=v) for k, v in ids.items()],
    )


def test_first_seen_records_baseline(tmp_path):
    reg = PartyRegistry.load(tmp_path / "kp.json")
    issues = reg.reconcile([_person("张三", "乙方", 身份证号="110101199001011234")], "docA")
    assert issues == []                       # 首见不报，只入库
    assert reg.dirty
    assert reg.get("张三")["身份证号"]["value"] == "110101199001011234"
    assert reg.get("张三")["身份证号"]["first_seen_doc"] == "docA"


def test_second_seen_consistent_no_issue(tmp_path):
    reg = PartyRegistry.load(tmp_path / "kp.json")
    reg.reconcile([_person("张三", "乙方", 身份证号="110101199001011234")], "docA")
    issues = reg.reconcile([_person("张三", "乙方", 身份证号="110101199001011234")], "docB")
    assert issues == []                       # 再见一致不报


def test_second_seen_conflict_reports_without_overwriting(tmp_path):
    reg = PartyRegistry.load(tmp_path / "kp.json")
    reg.reconcile([_person("张三", "乙方", 身份证号="110101199001011234")], "docA")
    # 末位被 OCR 读错 4→9
    issues = reg.reconcile([_person("张三", "乙方", 身份证号="110101199001011239")], "docB")
    assert len(issues) == 1
    assert issues[0].category == "identity"
    assert issues[0].item == "张三·身份证号"
    # 基准保持稳定，不被本次冲突值覆盖
    assert reg.get("张三")["身份证号"]["value"] == "110101199001011234"


def test_separator_noise_not_reported(tmp_path):
    reg = PartyRegistry.load(tmp_path / "kp.json")
    reg.reconcile([_person("张三", "乙方", 电话="139 1234 5678")], "docA")
    issues = reg.reconcile([_person("张三", "乙方", 电话="13912345678")], "docB")
    assert issues == []                       # 空格/分隔符差异不算冲突


def test_org_identifiers_also_checked(tmp_path):
    reg = PartyRegistry.load(tmp_path / "kp.json")
    reg.reconcile([_person("示例置业有限公司", "甲方", 银行账号="6222000011112222")], "docA")
    issues = reg.reconcile([_person("示例置业有限公司", "甲方", 银行账号="6222000011113333")], "docB")
    assert len(issues) == 1                   # 机构账号一视同仁核对
    assert issues[0].item == "示例置业有限公司·银行账号"


def test_set_and_remove(tmp_path):
    reg = PartyRegistry.load(tmp_path / "kp.json")
    reg.set("张三", "身份证号", "110101199001011234")
    assert reg.get("张三")["身份证号"]["first_seen_doc"] == "(manual)"
    assert reg.remove("张三", "身份证号") is True
    assert reg.get("张三") is None            # 标识删空后清掉主体空壳
    assert reg.remove("查无此人") is False


def test_save_load_roundtrip_and_perm(tmp_path):
    p = tmp_path / "kp.json"
    reg = PartyRegistry.load(p)
    reg.reconcile([_person("张三", "乙方", 身份证号="110101199001011234")], "docA")
    reg.save()
    assert oct(p.stat().st_mode)[-3:] == "600"   # PII 文件 0600
    reg2 = PartyRegistry.load(p)
    assert reg2.get("张三")["身份证号"]["value"] == "110101199001011234"


def test_load_missing_or_corrupt_returns_empty(tmp_path):
    assert PartyRegistry.load(tmp_path / "nope.json").all_parties() == {}
    bad = tmp_path / "bad.json"
    bad.write_text("not json{{", encoding="utf-8")
    assert PartyRegistry.load(bad).all_parties() == {}


def test_canon_strips_noise_keeps_digits():
    assert _canon(" 110101-1990 ") == "1101011990"
    assert _canon("139；138") == "139138"


# ---- 实体对齐：同实体不同名字归并到一个 key（治 known_parties 分裂）----
#
# 复现的生产分裂形态：同一公司被识别成"示例置业"与"示例奥业"（一字之差，幻觉/误读），
# 但共用同一章号；按字面 name 作 key 会分裂成两条，跨合同核对不到一起。
_FAKE_SEAL = "990011223344"   # 虚构章号占位


def test_strong_id_merge_unifies_variant_names(tmp_path):
    """同章号的两个名字变体 → 归并到首见 key，不新建第二个；本次名字记入别名表。"""
    reg = PartyRegistry.load(tmp_path / "kp.json")
    reg.reconcile([_person("示例置业有限公司", "甲方", 印章=_FAKE_SEAL)], "docA")
    issues = reg.reconcile([_person("示例奥业有限公司", "甲方", 印章=_FAKE_SEAL)], "docB")
    assert issues == []                                  # 同实体同章号，不报冲突
    assert reg.get("示例奥业有限公司") is None            # 不分裂出第二个 key
    assert reg.get("示例置业有限公司")["印章"]["value"] == _FAKE_SEAL
    assert reg._data["aliases"]["示例奥业有限公司"] == "示例置业有限公司"


def test_alias_resolves_later_weak_only_occurrence(tmp_path):
    """学到别名后，即便后续只带弱标识（电话）也能归到 canonical，不分裂。"""
    reg = PartyRegistry.load(tmp_path / "kp.json")
    reg.reconcile([_person("示例置业有限公司", "甲方", 印章=_FAKE_SEAL)], "docA")
    reg.reconcile([_person("示例奥业有限公司", "甲方", 印章=_FAKE_SEAL)], "docB")
    reg.reconcile([_person("示例奥业有限公司", "甲方", 电话="0571-88880000")], "docC")
    assert reg.get("示例奥业有限公司") is None            # 仍不分裂
    assert reg.get("示例置业有限公司")["电话"]["value"] == "0571-88880000"


def test_weak_identifier_does_not_merge_distinct_entities(tmp_path):
    """弱标识（开户行/电话）多人共用，绝不据此合并——否则会把不同实体焊死。"""
    reg = PartyRegistry.load(tmp_path / "kp.json")
    reg.reconcile([_person("示例置业有限公司", "甲方", 开户行="中国银行某支行")], "docA")
    reg.reconcile([_person("另一家置业有限公司", "甲方", 开户行="中国银行某支行")], "docB")
    assert reg.get("示例置业有限公司") is not None
    assert reg.get("另一家置业有限公司") is not None      # 两个实体各自独立
    assert reg._data["aliases"] == {}                     # 没有误并


def test_role_prefix_normalized_to_same_key(tmp_path):
    """称谓前缀差异归一：『甲方：示例置业』与『示例置业』归到同一 key。"""
    reg = PartyRegistry.load(tmp_path / "kp.json")
    reg.reconcile([_person("甲方：示例置业有限公司", "甲方", 银行账号="6222000011112222")], "docA")
    issues = reg.reconcile([_person("示例置业有限公司", "甲方", 银行账号="6222000011112222")], "docB")
    assert issues == []
    assert reg.get("示例置业有限公司")["银行账号"]["value"] == "6222000011112222"
    assert reg.get("甲方：示例置业有限公司") is None


def test_canon_name_separator_gated():
    """称谓前缀仅在后接分隔符时剥离；前缀恰为名字一部分（无分隔符）不误伤。"""
    assert _canon_name("甲方：示例置业有限公司") == "示例置业有限公司"
    assert _canon_name(" 出卖人  示例置业 ") == "示例置业"   # 空白也算分隔
    assert _canon_name("甲方物流有限公司") == "甲方物流有限公司"  # 无分隔符，不剥
    assert _canon_name("买方") == "买方"                      # 纯称谓无名字，原样保留


def test_is_strong_label():
    for lab in ("身份证号", "银行账号", "印章", "统一社会信用代码", "税号"):
        assert _is_strong_label(lab)
    for lab in ("电话", "开户行", "地址", "职位"):
        assert not _is_strong_label(lab)


# ---- 展示折叠：同值多 label 合并（治 party list/show 里同号被不同 label 重复堆叠）----


def test_group_by_value_merges_same_number_under_different_labels():
    """同一个号被两份文档写成『电话』『联系电话』→ 折叠成一行『电话/联系电话』，rec 取首个基准。"""
    ids = {
        "身份证号": {"value": "110101199001011234", "first_seen_doc": "docA"},
        "电话": {"value": "139 1234 5678", "first_seen_doc": "docA"},   # 带空格，canon 后与下行相等
        "联系电话": {"value": "13912345678", "first_seen_doc": "docB"},
    }
    rows = group_by_value(ids)
    assert [label for label, _ in rows] == ["身份证号", "电话/联系电话"]
    # 折叠组取首个出现的 rec（即基准首见那条），不被后见的 docB 顶掉
    assert dict(rows)["电话/联系电话"]["first_seen_doc"] == "docA"


def test_group_by_value_keeps_distinct_numbers_separate():
    """公司总机 vs 联系人线是两个真不同的号 → 各自独立，绝不因 label 近似而合并。"""
    ids = {
        "电话": {"value": "0571-88660000"},
        "联系电话": {"value": "0571-88880051"},
    }
    rows = group_by_value(ids)
    assert [label for label, _ in rows] == ["电话", "联系电话"]


def test_group_by_value_empty_and_order_preserved():
    assert group_by_value({}) == []
    # 各值互不相同时：原样、保持插入顺序，不动 label
    ids = {"身份证号": {"value": "A"}, "银行账号": {"value": "B"}, "电话": {"value": "C"}}
    assert [label for label, _ in group_by_value(ids)] == ["身份证号", "银行账号", "电话"]