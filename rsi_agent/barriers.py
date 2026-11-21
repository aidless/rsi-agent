"""信息屏障：密封私集 + 类型隔离。

注意：这是进程内的"构造性屏障 + 纪律屏障"，不是安全边界。密封靠
(a) 私集实例保存在模块级 _SEALED，实例本身没有指向 task 的属性；
(b) InnerAgent 只接受 PublicHistory，observe() 显式拒绝其它类型；
(c) public/private 使用不同 seed，private 仅在最终确认时调用一次。
测试用 gc.get_referents 与类型断言验证屏障，而不是靠口头声明。
"""
from __future__ import annotations

from .core import PublicHistory, ScoreVector
from .tasks import BinPackTask

_SEALED: dict[int, BinPackTask] = {}


class SealedPrivateSet:
    """Held-out 实例与 scorer；永不传给 InnerAgent。"""

    __slots__ = ("_id", "_evaluations")

    def __init__(self, task: BinPackTask) -> None:
        self._id = id(self)
        _SEALED[self._id] = task
        self._evaluations = 0

    def evaluate(self, policy) -> ScoreVector:
        self._evaluations += 1
        return _SEALED[self._id].evaluate(policy)

    @property
    def evaluation_count(self) -> int:
        return self._evaluations

    def __repr__(self) -> str:
        return "<SealedPrivateSet:SEALED>"


class BarrierViolation(TypeError):
    """InnerAgent 试图接触非公开评测数据。"""


def assert_public_only(history: object) -> None:
    if not isinstance(history, PublicHistory):
        raise BarrierViolation("InnerAgent 只能观察 PublicHistory，收到 " + type(history).__name__)
    for entry in history.entries:
        if not (isinstance(entry.policy_hash, str) and isinstance(entry.public_mean, float)):
            raise BarrierViolation("PublicHistory 中出现非公开字段")
