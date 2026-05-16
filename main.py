import argparse
import pandas as pd
import nids
import model
import torch
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Create a model with a CSV training dataset and CSV training labels and output a model_weights.pth file.", nargs=2, metavar=("dataset.csv", "labels.csv"))
    parser.add_argument("--test", help="Perform validation on a model with a CSV validation dataset and CSV validation labels, then print its accuracy.", nargs=2, metavar=("dataset.csv", "labels.csv"))
    parser.add_argument("--capture-file", help="Packet capture file containing network traffic to analyze.", metavar="capture.pcap")

    args = parser.parse_args()

    parser.print_help()

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    prediction_model = model.NeuralNetwork(input_features=5, output_features=2).to(device)

    try:
        prediction_model.load_state_dict(torch.load('model_weights.pth', weights_only=True, map_location=torch.device(device)))
    except FileNotFoundError:
        if len(sys.argv) == 1 or sys.argv[1] == "--test":
            print("ERROR: No model found. Place a model_weights.pth file in the current directory or generate a new one with the --train option.")
            sys.exit(1)

    if args.train and len(args.train) == 2:
        packet_capture_filename, labels_filename = args.train
        training_packet_df = pd.read_csv(packet_capture_filename)
        training_label_df = pd.read_csv(labels_filename)

        training_packet_df = nids.preprocess(training_packet_df)
        training_packet_dataset = nids.PacketDataset(training_packet_df, training_label_df)

        model.ModelTrainer(prediction_model, device).train(training_packet_dataset)

    if args.test and len(args.test) == 2:
        packet_capture_filename, labels_filename = args.test
        test_packet_df = pd.read_csv(packet_capture_filename)
        test_label_df = pd.read_csv(labels_filename)

        test_packet_df = nids.preprocess(test_packet_df)
        test_packet_dataset = nids.PacketDataset(test_packet_df, test_label_df)

        test_dataloader = torch.utils.data.DataLoader(test_packet_dataset, batch_size=64, num_workers=3, pin_memory=True)
        model.ModelTester(prediction_model, device).test_loop(test_dataloader)

    if not args.train and not args.test:
        nids.PacketSniffer(prediction_model, device, args.capture_file)

if __name__ == "__main__":
    main()
