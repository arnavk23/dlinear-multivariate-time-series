import unittest
import torch
from dlinear import DLinear

class TestDLinear(unittest.TestCase):
    def test_forward_shape(self):
        batch, seq_len, input_dim, pred_len = 4, 96, 8, 14
        model = DLinear(input_dim=input_dim, seq_len=seq_len, pred_len=pred_len)
        x = torch.randn(batch, seq_len, input_dim)
        out = model(x)
        self.assertEqual(out.shape, (batch, pred_len))

if __name__ == "__main__":
    unittest.main()
