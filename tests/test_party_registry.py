"""
PartyRegistry 首见入库 / 再见校对 行为测试。

PII 一律占位：张三 + 明显虚构的号码（110101199001011234 等），绝不用真实身份证。
"""
from contract_archive.archive.party_registry import PartyRegistry, _canon
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