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
        # Convert to float32 numpy to resolve BF16 errors and adapt to joblib caching
        matrix_np = tensor.detach().cpu().contiguous().to(torch.float32).numpy()
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
        """Test full-rank matrix (even distribution of knowledge)"""
        # Singular values of an identity matrix are all 1
        tensor = torch.eye(100)
        fingerprint = SVDAnalyzer.get_svd_fingerprint(tensor, bins=10)
        er = SVDAnalyzer.get_effective_rank(tensor)

        print(f"\nFull Rank [Eye]: {fingerprint} ER(Effective Rank): {er:.2f}")
        # Ideally, the fingerprint of a full-rank matrix should be fully filled [██████████]
        self.assertIn("█", fingerprint)
        self.assertAlmostEqual(er, 100.0, delta=1.0)

    def test_low_rank_matrix(self):
        """Test low-rank matrix (feature collapse)"""
        # Construct a 100x100 matrix with a rank of only 1
        tensor = torch.ones((100, 100))
        fingerprint = SVDAnalyzer.get_svd_fingerprint(tensor, bins=10)
        er = SVDAnalyzer.get_effective_rank(tensor)

        print(f"Low Rank [Ones]: {fingerprint} ER: {er:.2f}")
        # For a rank-1 matrix, only the first bin should be filled [█         ]
        self.assertTrue(fingerprint.startswith("[█"))
        self.assertLess(er, 5.0)

    def test_random_matrix(self):
        """Test random matrix (simulating layers in training)"""
        tensor = torch.randn(100, 100)
        fingerprint = SVDAnalyzer.get_svd_fingerprint(tensor, bins=10)
        er = SVDAnalyzer.get_effective_rank(tensor)

        print(f"Random Matrix:   {fingerprint} ER: {er:.2f}")
        # Random matrices usually exhibit smooth decay
        self.assertIsInstance(fingerprint, str)
        self.assertTrue(er > 10.0)

    def test_high_dim_tensor(self):
        """Test high-dimensional Tensor (simulating Conv2d or linear layer parameters)"""
        # Simulate shape [out_channels, in_channels, k, k]
        tensor = torch.randn(32, 64, 3, 3)
        fingerprint = SVDAnalyzer.get_svd_fingerprint(tensor, bins=10)

        print(f"4D Tensor:       {fingerprint}")
        self.assertEqual(len(fingerprint), 12)  # [ + 10 chars + ]

    def test_invalid_input(self):
        """Test invalid input"""
        # 1D tensors do not have a concept of singular values
        tensor = torch.randn(10)
        fingerprint = SVDAnalyzer.get_svd_fingerprint(tensor)
        self.assertEqual(fingerprint, "")


if __name__ == '__main__':
    unittest.main()
