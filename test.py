import unittest
import pandas as pd
import nids
import torch

class PacketDatasetTest(unittest.TestCase):
    def setUp(self):
        packet = [[1, 0.000000, "192.168.2.15", "192.168.100.5", "TLSv1", 1294, ""]]
        label = [[1, 0]]

        packet_df = pd.DataFrame(packet, columns=["No.", "Time", "Source", "Destination", "Protocol", "Length", "Info"])
        label_df = pd.DataFrame(label, columns=["No.", "x"])
        self.packet_dataset = nids.PacketDataset(packet_df, label_df)

    def test_get_number_of_packets(self):
        number_of_packets_result = len(self.packet_dataset)

        self.assertEqual(number_of_packets_result, 1)

    def test_get_packet(self):
        packet_length_column_number = 5
        destination_column_number = 3

        packet, label = self.packet_dataset[0]
        packet_length_result = packet[packet_length_column_number]
        destination_result = packet[destination_column_number]

        self.assertEqual((packet_length_result, destination_result, label), (1294, "192.168.100.5", 0))


class NeuralNetworkTest(unittest.TestCase):
    def setUp(self):
        self.number_of_samples = 128
        self.number_of_input_features = 7
        self.number_of_output_values = 2

        self.model = nids.NeuralNetwork(self.number_of_input_features, self.number_of_output_values)
        self.input_tensor = torch.zeros(self.number_of_samples, self.number_of_input_features)

    def test_canary(self):
        self.assertEqual(True, True)

    def test_output_shape(self):
        output_tensor = self.model(self.input_tensor)

        output_tensor_shape = output_tensor.shape

        self.assertEqual(output_tensor_shape, (self.number_of_samples, self.number_of_output_values))

if __name__ == '__main__':
    unittest.main()
