import unittest
import nids

class PacketDatasetTest(unittest.TestCase):
    def setUp(self):
        self.max_number_of_packets = 100
        self.packet_dataset = nids.PacketDataset("out.csv", "labels.csv", self.max_number_of_packets)

    def test_get_number_of_packets(self):
        number_of_packets_result = len(self.packet_dataset)

        self.assertEqual(number_of_packets_result, self.max_number_of_packets)

    def test_get_packet_length(self):
        packet_length = 1294

        packet_length_result = self.packet_dataset[0]["Length"]

        self.assertEqual(packet_length_result, packet_length)

    def test_get_packet_destination(self):
        destination = "192.168.100.5"

        destination_result = self.packet_dataset[0]["Destination"]

        self.assertEqual(destination_result, destination)

    def test_get_packet_label(self):
        label = 0

        label_result = self.packet_dataset[0]["x"]

        self.assertEqual(label_result, label)

if __name__ == '__main__':
    unittest.main()
