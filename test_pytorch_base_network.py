import unittest
import numpy as np
import torch

from pytorch_base_network import PyTorchBaseNetwork


class TestPyTorchBaseNetwork(unittest.TestCase):
    # verify the types that forward accepts
    def test_single_tensor(self):
        nn = PyTorchBaseNetwork(input_shape=(5,), lin_layers=[100, 100], output_shape=(4,))
        ten = torch.tensor([1, 2, 3, 4, 5]).float()
        out = nn.forward(ten)
        self.assertTrue(isinstance(nn.forward(ten), torch.FloatTensor))
        self.assertTupleEqual(out.shape, (4,))

    def test_list_tensors(self):
        nn = PyTorchBaseNetwork(input_shape=(5,), lin_layers=[100, 100], output_shape=(4,))
        ten = torch.tensor([[1, 2, 3, 4, 5], [1, 4, 2, 5, 6]]).float()
        out = nn.forward(ten)
        self.assertTrue(isinstance(nn.forward(ten), torch.FloatTensor))
        self.assertTupleEqual(out.shape, (2, 4))

    def test_free_tuples(self):
        nn = PyTorchBaseNetwork(input_shape=(5,), lin_layers=[100, 100], output_shape=(4,))
        ten = torch.tensor(([1, 2, 3, 4, 5], (1, 4, 2, 5, 6))).float()
        out = nn.forward(ten)
        self.assertTrue(isinstance(nn.forward(ten), torch.FloatTensor))
        self.assertTupleEqual(out.shape, (2, 4))

    def test_np_array_fails(self):
        nn = PyTorchBaseNetwork(input_shape=(5,), lin_layers=[100, 100], output_shape=(4,))
        ten = np.array(([1, 2, 3, 4, 5], (1, 4, 2, 5, 6))).astype(float)
        with self.assertRaises(TypeError):
            nn.forward(ten)

    def test_tensors_double_fails(self):
        nn = PyTorchBaseNetwork(input_shape=(5,), lin_layers=[100, 100], output_shape=(4,))
        ten = torch.tensor(([1, 2, 3, 4, 5], (1, 4, 2, 5, 6))).double()
        with self.assertRaises(TypeError):
            nn.forward(ten)

    def test_tensors_int_fails(self):
        nn = PyTorchBaseNetwork(input_shape=(5,), lin_layers=[100, 100], output_shape=(4,))
        ten = torch.tensor(([1, 2, 3, 4, 5], (1, 4, 2, 5, 6)))
        with self.assertRaises(TypeError):
            nn.forward(ten)

    def test_nd_tensors(self):
        nn = PyTorchBaseNetwork(input_shape=(5,), lin_layers=[100, 100], output_shape=(4,))
        ten = torch.tensor([[[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [1, 4, 2, 5, 6]],
                            [[1, 2, 3, 4, 5], [1, 4, 2, 5, 6], [2, 3, 4, 5, 6]]]).float()
        out = nn.forward(ten)
        self.assertTrue(isinstance(nn.forward(ten), torch.FloatTensor))
        self.assertTupleEqual(out.shape, (2, 3, 4))

    #verify save and load behavior
    def test_loadsave_model(self):
        nn = PyTorchBaseNetwork(input_shape=(2,), lin_layers=[4, 4], output_shape=(2,), dropout_p=0.1)
        nn.train()
        true_fun = lambda x : torch.tensor(np.abs(np.matmul(x.numpy(), np.array(([1, 2], [-3, 1])))))
        optimizer = torch.optim.Adam(nn.parameters())
        for i in range(10000):
            batch=torch.randn(100, 2)
            target = true_fun(batch)
            loss=torch.nn.MSELoss()(target.float(), nn.forward(batch))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%1000==0:
                print(nn.state_dict())

        nn.eval()
        test_batch=torch.randn(10, 2)
        print(true_fun(test_batch))
        print(nn.forward(test_batch))

        nn.save_model("saved.ckp", "trivial network")

        nn2 = PyTorchBaseNetwork("saved.ckp")
        test_batch=torch.randn(10, 2)
        print(true_fun(test_batch))
        print(nn.forward(test_batch))






if __name__ == '__main__':
    unittest.main()
