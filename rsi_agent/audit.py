"""哈希链审计日志：tamper-evident，任何人可离线重放验证。

每条记录 = {seq, prev_hash, ts_tag, kind, payload}，hash = sha256(prev_hash || canonical(payload_no_hash))。
防篡改三件事都能被 verify() 抓到：改内容（hash 不匹配）、删记录（prev_hash 断链/seq 跳号）、
换顺序（prev_hash 对不上）。genesis 的 prev_hash = "0"*64 是公开常量。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .core import canonical_hash

GENESIS = "0" * 64


@dataclass
class AuditRecord:
    seq: int
    prev_hash: str
    kind: str
    payload: dict
    ts_tag: str
    hash: str = ""

    def body(self) -> dict:
        return {"seq": self.seq, "prev_hash": self.prev_hash, "kind": self.kind,
                "payload": self.payload, "ts_tag": self.ts_tag}


class AuditError(ValueError):
    pass


class AuditLog:
    def __init__(self) -> None:
        self._records: list[AuditRecord] = []
        self._seal_record = AuditRecord(seq=-1, prev_hash="", kind="", payload={}, ts_tag="")

    def append(self, kind: str, payload: dict, ts_tag: str = "") -> AuditRecord:
        prev = self._records[-1].hash if self._records else GENESIS
        rec = AuditRecord(seq=len(self._records), prev_hash=prev, kind=kind,
                          payload=payload, ts_tag=ts_tag)
        rec.hash = canonical_hash(rec.body())
        self._records.append(rec)
        return rec

    @property
    def records(self) -> tuple[AuditRecord, ...]:
        return tuple(self._records)

    def verify(self) -> tuple[bool, str]:
        """返回 (ok, detail)。任一条失配即 fail-closed。"""
        expected_prev = GENESIS
        for i, rec in enumerate(self._records):
            if rec.seq != i:
                return False, f"SEQ_MISMATCH at index {i}: seq={rec.seq}"
            if rec.prev_hash != expected_prev:
                return False, f"CHAIN_BREAK at index {i}: prev={rec.prev_hash[:12]}"
            if rec.hash != canonical_hash(rec.body()):
                return False, f"HASH_MISMATCH at index {i}: kind={rec.kind}"
            expected_prev = rec.hash
        return True, f"OK {len(self._records)} records"

    def to_lines(self) -> list[str]:
        return [__import__("json").dumps({
            "seq": r.seq, "prev_hash": r.prev_hash, "kind": r.kind,
            "payload": r.payload, "ts_tag": r.ts_tag, "hash": r.hash,
        }, ensure_ascii=False, sort_keys=True) for r in self._records]
