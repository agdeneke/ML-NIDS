import unittest
import pandas as pd
import features
import model
import torch


class PacketDatasetTest(unittest.TestCase):
    def setUp(self):
        self.packet = [[1, 0, 1294]]
        self.label = [[1, 0]]

        packet_df = pd.DataFrame(self.packet, columns=["No.", "Time", "Length"])
        label_df = pd.DataFrame(self.label, columns=["No.", "x"])
        self.packet_dataset = model.PacketDataset(packet_df, label_df)

    def test_get_number_of_packets(self):
        number_of_packets_result = len(self.packet_dataset)

        self.assertEqual(number_of_packets_result, 1)

    def test_get_packet(self):
        packet_result, label_result = self.packet_dataset[0]

        self.assertEqual(packet_result.tolist(), self.packet[0])


class NeuralNetworkTest(unittest.TestCase):
    def setUp(self):
        self.number_of_samples = 128
        self.number_of_input_features = 5
        self.number_of_output_values = 2

        self.model = model.NeuralNetwork(self.number_of_input_features, self.number_of_output_values)
        self.input_tensor = torch.zeros(self.number_of_samples, self.number_of_input_features)

    def test_canary(self):
        self.assertEqual(True, True)

    def test_output_shape(self):
        output_tensor = self.model(self.input_tensor)

        output_tensor_shape = output_tensor.shape

        self.assertEqual(output_tensor_shape, (self.number_of_samples, self.number_of_output_values))


class FeatureCreationTest(unittest.TestCase):
    def setUp(self):
        self.packets = [[0.0, "192.168.1.254", 50, 1.0, 1.0],
                        [0.5, "192.168.1.254", 50, 1.0, 1.0]]

    def test_find_arp_request_rate(self):
        arp_request_rate = features.find_arp_request_rate("192.168.1.254", self.packets)

        self.assertEqual(arp_request_rate, 1)

    def test_find_tcp_rate(self):
        tcp_rate = features.find_tcp_request_rate(self.packets)

        self.assertEqual(tcp_rate, 1)


if __name__ == '__main__':
    unittest.main()
