import unittest
import pandas as pd
import numpy as np
import nids
import model
import torch

class PacketDatasetTest(unittest.TestCase):
    def setUp(self):
        self.packet = [[1, 0, 1294]]
        self.label = [[1, 0]]

        packet_df = pd.DataFrame(self.packet, columns=["No.", "Time", "Length"])
        label_df = pd.DataFrame(self.label, columns=["No.", "x"])
        self.packet_dataset = nids.PacketDataset(packet_df, label_df)

    def test_get_number_of_packets(self):
        number_of_packets_result = len(self.packet_dataset)

        self.assertEqual(number_of_packets_result, 1)

    def test_get_packet(self):
        packet_result, label_result = self.packet_dataset[0]

        self.assertEqual(packet_result.tolist(), self.packet[0])

class NeuralNetworkTest(unittest.TestCase):
    def setUp(self):
        self.number_of_samples = 128
        self.number_of_input_features = 7
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
        self.packet_df = pd.DataFrame([[pd.to_datetime(0), "192.168.1.254", 1.0, 1.0]], columns=["Time", "Source", "Protocol_ARP", "Protocol_TCP"])

    def test_find_arp_request_rate(self):
        packet_df_with_arp_request_rate = nids.find_arp_request_rate(self.packet_df)

        self.assertEqual(packet_df_with_arp_request_rate["arp_request_rate"][0], 1)

    def test_find_tcp_rate(self):
        packet_df_with_tcp_rate = nids.find_tcp_rate(self.packet_df)

        self.assertEqual(packet_df_with_tcp_rate["tcp_rate"][0], 1)

if __name__ == '__main__':
    unittest.main()
