import unittest
import nids

class PacketDatasetTest(unittest.TestCase):
    def setUp(self):
        self.packet_dataset = nids.PacketDataset("out.csv")

    def test_get_number_of_packets(self):
        number_of_packets = 38

        number_of_packets_result = len(self.packet_dataset)

        self.assertEqual(number_of_packets_result, number_of_packets)

    def test_get_packet_length(self):
        packet_length = 175

        packet_length_result = self.packet_dataset[3]["length"]

        self.assertEqual(packet_length_result, packet_length)

    def test_get_packet_destination_mac(self):
        destination_mac = "ff:ff:ff:ff:ff:ff"

        destination_mac_result = self.packet_dataset[0]["destination_mac"]

        self.assertEqual(destination_mac_result, destination_mac)

if __name__ == '__main__':
    unittest.main()
