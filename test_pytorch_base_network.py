import unittest

import numpy as np
import torch

from pytorch_base_network import PyTorchBaseNetwork

def numpy_mse(a,b):
    return np.mean((a- b) ** 2)

def true_fun(x):
    return np.abs(np.matmul(x, np.array(([1, 2], [-3, 1]))))

class TestPyTorchBaseNetwork(unittest.TestCase):
    # verify the types that forward accepts
    def test_single_tensor(self):
        pyt_net = PyTorchBaseNetwork(input_shape=(5,), lin_layers=[100, 100], output_shape=(4,))
        ten = torch.tensor([1, 2, 3, 4, 5]).float()
        out = pyt_net.forward(ten)
        self.assertTrue(isinstance(pyt_net.forward(ten), torch.FloatTensor))
        self.assertTupleEqual(out.shape, (4,))

    def test_list_tensors(self):
        pyt_net = PyTorchBaseNetwork(input_shape=(5,), lin_layers=[100, 100], output_shape=(4,))
        ten = torch.tensor([[1, 2, 3, 4, 5], [1, 4, 2, 5, 6]]).float()
        out = pyt_net.forward(ten)
        self.assertTrue(isinstance(pyt_net.forward(ten), torch.FloatTensor))
        self.assertTupleEqual(out.shape, (2, 4))

    def test_free_tuples(self):
        pyt_net = PyTorchBaseNetwork(input_shape=(5,), lin_layers=[100, 100], output_shape=(4,))
        ten = torch.tensor(([1, 2, 3, 4, 5], (1, 4, 2, 5, 6))).float()
        out = pyt_net.forward(ten)
        self.assertTrue(isinstance(pyt_net.forward(ten), torch.FloatTensor))
        self.assertTupleEqual(out.shape, (2, 4))

    def test_np_array_fails(self):
        pyt_net = PyTorchBaseNetwork(input_shape=(5,), lin_layers=[100, 100], output_shape=(4,))
        ten = np.array(([1, 2, 3, 4, 5], (1, 4, 2, 5, 6))).astype(float)
        with self.assertRaises(TypeError):
            pyt_net.forward(ten)

    def test_tensors_double_fails(self):
        pyt_net = PyTorchBaseNetwork(input_shape=(5,), lin_layers=[100, 100], output_shape=(4,))
        ten = torch.tensor(([1, 2, 3, 4, 5], (1, 4, 2, 5, 6))).double()
        with self.assertRaises(TypeError):
            pyt_net.forward(ten)

    def test_tensors_int_fails(self):
        pyt_net = PyTorchBaseNetwork(input_shape=(5,), lin_layers=[100, 100], output_shape=(4,))
        ten = torch.tensor(([1, 2, 3, 4, 5], (1, 4, 2, 5, 6)))
        with self.assertRaises(TypeError):
            pyt_net.forward(ten)

    def test_nd_tensors(self):
        pyt_net = PyTorchBaseNetwork(input_shape=(5,), lin_layers=[100, 100], output_shape=(4,))
        ten = torch.tensor([[[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [1, 4, 2, 5, 6]],
                            [[1, 2, 3, 4, 5], [1, 4, 2, 5, 6], [2, 3, 4, 5, 6]]]).float()
        out = pyt_net.forward(ten)
        self.assertTrue(isinstance(pyt_net.forward(ten), torch.FloatTensor))
        self.assertTupleEqual(out.shape, (2, 3, 4))

    # verify trainint and save load behavior
    def test_loadsave_model(self):
        pyt_net = PyTorchBaseNetwork(input_shape=(2,), lin_layers=[4, 4], output_shape=(2,), dropout_p=0.2)
        pyt_net.train()

        optimizer = torch.optim.Adam(pyt_net.parameters())
        cum_loss = 0
        old_loss = float('inf')
        print("Training a network for regression")

        # simulate epochs (actually all epochs get new random samples)
        epochs = 3
        batches_in_epoch = 500
        samples_in_train_batch = 100
        for i in range(epochs):
            for j in range(batches_in_epoch):
                batch = torch.randn(samples_in_train_batch, 2)
                target = torch.tensor(true_fun(batch.data.numpy()))
                loss = torch.nn.MSELoss()(target.float(), pyt_net.forward(batch))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cum_loss = cum_loss + loss.data.numpy()
            print("average loss for training epoch " + str(i) + " = " + str(cum_loss / batches_in_epoch))
            self.assertGreater(old_loss, cum_loss)
            old_loss = cum_loss
            cum_loss = 0

        # evaluation
        samples_in_test_batch = 10
        test_batch = np.random.randn(samples_in_test_batch, 2)
        with torch.no_grad():
            # set to test mode
            pyt_net.eval()

            # verify that 2 runs of forward on same data and network give same results
            est_nn = pyt_net.forward(torch.tensor(test_batch).float())
            loss_test_nn = numpy_mse(true_fun(test_batch), est_nn.data.numpy())
            print("average loss on test with original network:" + str(loss_test_nn))

            est_nn_retry = pyt_net.forward(torch.tensor(test_batch).float())
            loss_test_nn_retry = numpy_mse(true_fun(test_batch), est_nn.data.numpy())
            print("average loss on test with original network, 2nd try:" + str(loss_test_nn))

            self.assertAlmostEqual(numpy_mse(est_nn.data.numpy(),est_nn_retry.data.numpy()), 0.0)
            self.assertAlmostEqual(loss_test_nn, loss_test_nn_retry)

            pyt_net.save_model("mod_test.ckp" , "regression model for test")

            # verify that 2 runs of forward on same data and network give same results

            #load the model in a new network and set to evaluation mode
            pyt_net_loaded = PyTorchBaseNetwork("mod_test.ckp")
            pyt_net_loaded.eval()

            est_nn_loaded = pyt_net_loaded.forward(torch.tensor(test_batch).float())
            loss_test_nn_loaded = numpy_mse(true_fun(test_batch), est_nn_loaded.data.numpy())
            print("average loss on test with loaded network:" + str(loss_test_nn_loaded))

            self.assertAlmostEqual(numpy_mse(est_nn.data.numpy(),est_nn_loaded.data.numpy()), 0.0)
            self.assertAlmostEqual(loss_test_nn, loss_test_nn_loaded)


if __name__ == '__main__':
    unittest.main()
