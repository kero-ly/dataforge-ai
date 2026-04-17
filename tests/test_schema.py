# tests/test_schema.py
from dataforge.schema import DataRecord, RecordStatus


def test_datarecord_defaults():
    record = DataRecord(seed_data={"instruction": "hello"})
    assert record.id is not None
    assert record.status == RecordStatus.PENDING
    assert record.synthetic_data is None
    assert record.score is None
    assert record.metadata == {}


def test_datarecord_status_transition():
    record = DataRecord(seed_data={"instruction": "test"})
    record.status = RecordStatus.GENERATED
    assert record.status == RecordStatus.GENERATED


def test_datarecord_serialization_roundtrip():
    record = DataRecord(
        seed_data={"instruction": "test"},
        metadata={"model": "gpt-4o", "tokens": 100},
    )
    json_str = record.model_dump_json()
    restored = DataRecord.model_validate_json(json_str)
    assert restored.id == record.id
    assert restored.seed_data == record.seed_data
