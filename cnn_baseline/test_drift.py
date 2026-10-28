"""Numerical/segment contracts for the drift experiment; no real training."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import unittest

import numpy as np
import torch

from .losses import velocity_bias_loss
from .sequence_dataset import partition_segments


class DriftContracts(unittest.TestCase):
    def test_zero_residual(self):
        pred = torch.ones(2, 64, 2, requires_grad=True)
        total, velocity, bias = velocity_bias_loss(pred, pred.detach(), torch.full((2,64), .05))
        self.assertEqual(total.item(), 0.)
        total.backward(); torch.testing.assert_close(pred.grad, torch.zeros_like(pred))

    def test_constant_bias_not_scaled_by_segment_duration(self):
        pred = torch.tensor([.05, -.03]).expand(2,64,2).clone()
        for delta in (torch.full((2,64), .05), torch.linspace(.01,.1,64).expand(2,-1)):
            total, velocity, bias = velocity_bias_loss(pred, torch.zeros_like(pred), delta)
            torch.testing.assert_close(bias, velocity)
            torch.testing.assert_close(total, 2*velocity)

    def test_alternating_errors_cancel_but_keep_velocity_loss(self):
        pred = torch.ones(1,64,2)*.05; pred[:,1::2] *= -1
        total, velocity, bias = velocity_bias_loss(pred, torch.zeros_like(pred), torch.full((1,64),.05))
        self.assertGreater(velocity.item(), 0)
        torch.testing.assert_close(bias, torch.tensor(0.))
        torch.testing.assert_close(total, velocity)

    def test_prefixes_detect_error_hidden_by_endpoint_cancellation(self):
        pred = torch.ones(1,64,2)*.05; pred[:,32:] *= -1
        _, velocity, bias = velocity_bias_loss(pred, torch.zeros_like(pred), torch.full((1,64), .05))
        torch.testing.assert_close(bias, velocity*(2/3))

    def test_lambda_zero_has_exact_mse_value_and_gradient(self):
        torch.manual_seed(42)
        pred = torch.randn(2,64,2, requires_grad=True); target = torch.randn_like(pred)
        loss, _, _ = velocity_bias_loss(pred, target, torch.full((2,64),.05), weight=0)
        expected = torch.nn.functional.mse_loss(pred,target)
        torch.testing.assert_close(loss,expected,rtol=0,atol=0)
        grad, = torch.autograd.grad(loss,pred,retain_graph=True)
        reference, = torch.autograd.grad(expected,pred)
        torch.testing.assert_close(grad,reference,rtol=0,atol=0)

    def test_segments_do_not_bridge_sequences_or_filtered_gaps(self):
        index = [(0,i) for i in range(0,1000,10)]
        index += [(0,i) for i in range(1100,2100,10)]
        index += [(1,i) for i in range(0,800,10)]
        blocks, counts = partition_segments(index,64,10)
        self.assertEqual(blocks.shape,(3,64))
        self.assertEqual(counts['dropped_run_tail_windows'],88)
        mapped = np.array(index)[blocks]
        self.assertTrue(np.all(mapped[:,:,0] == mapped[:,:1,0]))
        self.assertTrue(np.all(np.diff(mapped[:,:,1],axis=1) == 10))
        self.assertEqual(np.unique(blocks).size,192)

    def test_invalid_inputs_rejected(self):
        pred=torch.ones(2,64,2); delta=torch.ones(2,64)
        for invalid in (torch.zeros_like(delta),-delta,delta*float('nan')):
            with self.assertRaises(ValueError):velocity_bias_loss(pred,pred,invalid)
        with self.assertRaises(ValueError):velocity_bias_loss(pred,pred,delta,weight=-1)
        with self.assertRaises(ValueError):velocity_bias_loss(pred,pred,delta,horizons=(65,))
        with self.assertRaises(ValueError):partition_segments([(0,0)],64,10)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_deterministic_cuda_backward(self):
        from .model_cnn_gru import CNNGRU
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(42)
        model=CNNGRU().cuda()
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        x=torch.randn(128,6,200,device='cuda')
        target=torch.zeros(2,64,2,device='cuda');delta=torch.full((2,64),.05,device='cuda')
        before=model.gru.weight_ih_l0.detach().clone()
        for weight in (0.,1.):
            optimizer.zero_grad()
            loss,_,_=velocity_bias_loss(model(x).reshape(2,64,2),target,delta,weight=weight)
            loss.backward()
            self.assertTrue(torch.isfinite(model.gru.weight_ih_l0.grad).all())
            optimizer.step()
        self.assertFalse(torch.equal(before,model.gru.weight_ih_l0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
