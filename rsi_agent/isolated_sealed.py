r"""进程隔离密封私集：私有 task 只在 worker 子进程内构造。

与进程内版 SealedPrivateSet 的接口兼容（evaluate / evaluation_count / __repr__），
但私有集的 BinPackTask 实例永不进主进程内存——主进程只经 Pipe 传 policy、
收 ScoreVector。worker 用 fork 启动（Linux 默认），是真正的 OS 级进程，私集
数据不在主进程地址空间，主进程 gc 扫不到。

这是对"进程内屏障"的升级：从构造性+纪律屏障升级为进程隔离。
"""
from __future__ import annotations

import multiprocessing as _mp

from .core import ScoreVector


def _sealed_worker(conn, seed: int) -> None:
    """worker 进程：构造私有集，循环收权重 tuple 求值并回送分数向量。

    只收/发纯数值元组，绝不传 task 的任何字段（真值、seed、样本），也不跨
    进程传对象（跨进程序列化会让 dataclass 退化，见 evaluate 注释）。
    conn 是 socketpair 的子进程端：主进程从另一端 send/recv。
    """
    from .core import HeuristicPolicy
    from .tasks import BinPackTask  # 在子进程内构造，主进程不持有该实例

    task = BinPackTask(seed=seed)
    while True:
        weights = conn.recv()
        if weights is None:  # 关闭信号
            break
        conn.send(task.evaluate(HeuristicPolicy(weights)).values)
    conn.close()


class IsolatedSealedPrivateSet:
    """与 SealedPrivateSet 接口兼容的进程隔离版。"""

    __slots__ = ("_parent", "_proc", "_evaluations")

    def __init__(self, seed: int) -> None:
        ctx = _mp.get_context("fork")
        self._parent, child = ctx.Pipe()
        self._proc = ctx.Process(
            target=_sealed_worker, args=(child, seed), daemon=True
        )
        self._proc.start()
        child.close()  # 主进程关掉子进程端副本，只留 self._parent 双向收发
        self._evaluations = 0

    def evaluate(self, policy) -> ScoreVector:
        # 只发纯权重 tuple：跨进程传 dataclass 对象会在子进程退化为 tuple，
        # 且只传公开权重本就是隔离要的语义（不把 InnerAgent 的对象送进私集进程）。
        self._parent.send(tuple(policy.weights))
        values = self._parent.recv()
        self._evaluations += 1
        return ScoreVector(tuple(values))

    @property
    def evaluation_count(self) -> int:
        return self._evaluations

    def close(self) -> None:
        if self._proc.is_alive():
            try:
                self._parent.send(None)
            except (BrokenPipeError, OSError):
                pass
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.terminate()
        self._parent.close()

    def __repr__(self) -> str:
        return "<SealedPrivateSet:SEALED-ISOLATED>"
