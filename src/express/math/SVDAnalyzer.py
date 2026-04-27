import os
import unittest

import numpy as np
import torch
from joblib import Memory

cachedir = '.cache'
memory = Memory(cachedir, verbose=int(os.getenv("JOBLIB_CACHE_DEBUG_LEVEL", 0)))


class SVDAnalyzer:
    @classmethod
    def get_svd_fingerprint(cls, tensor: torch.Tensor, bins: int = 10) -> str:
        if tensor.ndim < 2: return ""
        # 统一转为 float32 的 numpy，解决 BF16 报错并适配 joblib 缓存
        matrix_np = tensor.detach().to(torch.float32).cpu().numpy()
        return cls._cached_fingerprint_calc(matrix_np, bins)

    @staticmethod
    @memory.cache
    def _cached_fingerprint_calc(matrix_np: np.ndarray, bins: int) -> str:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        matrix = torch.from_numpy(matrix_np).to(device).float()
        if matrix.ndim > 2:
            matrix = matrix.flatten(start_dim=1)

        try:
            s = torch.linalg.svdvals(matrix)
            s = s / (s.max() + 1e-9)

            chars = " ▂▃▄▅▆▇█"
            num_chars = len(chars)
            chunks = torch.chunk(s, bins)

            fingerprint = "".join([
                chars[min(int(torch.mean(c).item() * num_chars), num_chars - 1)]
                for c in chunks
            ])
            return f"[{fingerprint}]"
        except Exception:
            return "[Error]"
        finally:
            del matrix
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @classmethod
    def get_effective_rank(cls, tensor: torch.Tensor) -> float:
        """
        修改点：同样采用缓存机制，大幅提升二次分析速度
        """
        if tensor.ndim < 2: return 0.0
        matrix_np = tensor.detach().to(torch.float32).cpu().numpy()
        return cls._cached_er_calc(matrix_np)

    @staticmethod
    @memory.cache
    def _cached_er_calc(matrix_np: np.ndarray) -> float:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        matrix = torch.from_numpy(matrix_np).to(device).float()
        if matrix.ndim > 2:
            matrix = matrix.flatten(start_dim=1)

        try:
            s = torch.linalg.svdvals(matrix)
            p = s / (s.sum() + 1e-9)
            entropy = -torch.sum(p * torch.log(p + 1e-9))
            return torch.exp(entropy).item()
        except Exception:
            return 0.0

class TestSVDAnalyzer(unittest.TestCase):

    def test_full_rank_matrix(self):
        """测试全秩矩阵（知识分布均匀）"""
        # 单位矩阵的奇异值全是 1
        tensor = torch.eye(100)
        fingerprint = SVDAnalyzer.get_svd_fingerprint(tensor, bins=10)
        er = SVDAnalyzer.get_effective_rank(tensor)

        print(f"\nFull Rank [Eye]: {fingerprint} ER(Effective Rank): {er:.2f}")
        # 理想状态下，全秩矩阵的指纹应该是满格的 [██████████]
        self.assertIn("█", fingerprint)
        self.assertAlmostEqual(er, 100.0, delta=1.0)

    def test_low_rank_matrix(self):
        """测试低秩矩阵（特征坍缩）"""
        # 构造一个 100x100 但秩仅为 1 的矩阵
        tensor = torch.ones((100, 100))
        fingerprint = SVDAnalyzer.get_svd_fingerprint(tensor, bins=10)
        er = SVDAnalyzer.get_effective_rank(tensor)

        print(f"Low Rank [Ones]: {fingerprint} ER: {er:.2f}")
        # 秩为 1 的矩阵，除了第一格，后面应该基本是空的 [█         ]
        self.assertTrue(fingerprint.startswith("[█"))
        self.assertLess(er, 5.0)

    def test_random_matrix(self):
        """测试随机矩阵（模拟训练中的层）"""
        tensor = torch.randn(100, 100)
        fingerprint = SVDAnalyzer.get_svd_fingerprint(tensor, bins=10)
        er = SVDAnalyzer.get_effective_rank(tensor)

        print(f"Random Matrix:   {fingerprint} ER: {er:.2f}")
        # 随机矩阵通常呈现平滑衰减
        self.assertIsInstance(fingerprint, str)
        self.assertTrue(er > 10.0)

    def test_high_dim_tensor(self):
        """测试多维 Tensor（模拟 Conv2d 或线性层参数）"""
        # 模拟 shape [out_channels, in_channels, k, k]
        tensor = torch.randn(32, 64, 3, 3)
        fingerprint = SVDAnalyzer.get_svd_fingerprint(tensor, bins=10)

        print(f"4D Tensor:       {fingerprint}")
        self.assertEqual(len(fingerprint), 12)  # [ + 10 chars + ]

    def test_invalid_input(self):
        """测试非法输入"""
        # 1D tensor 没有奇异值概念
        tensor = torch.randn(10)
        fingerprint = SVDAnalyzer.get_svd_fingerprint(tensor)
        self.assertEqual(fingerprint, "")


if __name__ == '__main__':
    unittest.main()
