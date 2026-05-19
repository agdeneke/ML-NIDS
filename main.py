import argparse
import pandas as pd
import nids
import model
import torch
import sys

def load_packet_dataset(packet_capture_filename: str, labels_filename: str) -> nids.PacketDataset:
    packet_df = pd.read_csv(packet_capture_filename)
    label_df = pd.read_csv(labels_filename)

    packet_df.preprocess(training_packet_df)
    return nids.PacketDataset(packet_df, label_df)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Create a model with a CSV training dataset and CSV training labels and output a model_weights.pth file.", nargs=2, metavar=("dataset.csv", "labels.csv"))
    parser.add_argument("--test", help="Perform validation on a model with a CSV validation dataset and CSV validation labels, then print its accuracy.", nargs=2, metavar=("dataset.csv", "labels.csv"))
    parser.add_argument("--capture-file", help="Packet capture file containing network traffic to analyze.", metavar="capture.pcap")
    parser.add_argument("--model-file", help="File where model is located in.", default="model_weights.pth", metavar="model_weights.pth")

    args = parser.parse_args()

    parser.print_help()

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    prediction_model = model.NeuralNetwork(input_features=5, output_features=2).to(device)

    try:
        prediction_model.load_state_dict(torch.load(args.model_file, weights_only=True, map_location=torch.device(device)))
    except FileNotFoundError:
        if not args.train:
            print(f"ERROR: No model found in {args.model_file}. Place a model file in the current directory or generate a new one with the --train option.")
            sys.exit(1)

    if args.train:
        dataset = load_packet_dataset(*args.train)
        model.ModelTrainer(prediction_model, device, args.model_file).train(dataset)

    if args.test:
        dataset = load_packet_dataset(*args.test)
        test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=3, pin_memory=True)
        model.ModelTester(prediction_model, device).test_loop(test_dataloader)

    if not args.train and not args.test:
        nids.PacketSniffer(prediction_model, device, args.capture_file)

if __name__ == "__main__":
    main()
