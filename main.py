import argparse
import pandas as pd
import sniffer
import features
import model
import torch
import sys
from app import create_app
from config import MODEL_CONFIG


def load_packet_dataset(packet_capture_filename: str,
                        labels_filename: str) -> model.PacketDataset:
    packets_df = pd.read_csv(packet_capture_filename)
    labels_df = pd.read_csv(labels_filename)

    packets_df = features.preprocess(packets_df)
    print(packets_df)
    return model.PacketDataset(packets_df, labels_df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",
                        help="Create a model with a CSV training dataset and C"
                        "SV training labels and output a model weights file.",
                        nargs=2, metavar=("dataset.csv", "labels.csv"))
    parser.add_argument("--test",
                        help="Perform validation on a model with a CSV validat"
                        "ion dataset and CSV validation labels, then print its"
                        "accuracy.",
                        nargs=2, metavar=("dataset.csv", "labels.csv"))
    parser.add_argument("--capture-file",
                        help="Packet capture file with network traffic to scan.",
                        metavar="capture.pcap")
    parser.add_argument("--host",
                        help="IP address to host a web dashboard for alerts.",
                        default="127.0.0.1", metavar="127.0.0.1")
    parser.add_argument("--port",
                        help="Port to host a web dashboard for alerts.",
                        default="5000", metavar="5000")
    parser.add_argument("--model-file",
                        help="File where model is located in.",
                        default="model_weights.pth", metavar="model_weights.pth")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    device = (torch.accelerator.current_accelerator().type
              if torch.accelerator.is_available() else "cpu")
    print(f"Using {device} device")

    prediction_model = model.NeuralNetwork(input_features=MODEL_CONFIG["input_features"],
                                           output_features=MODEL_CONFIG["output_features"]).to(device)

    try:
        prediction_model.load_state_dict(torch.load(args.model_file,
                                                    weights_only=True,
                                                    map_location=torch.device(device)))
    except FileNotFoundError:
        if not args.train:
            print(f"ERROR: No model found in {args.model_file}. Place a model "
                  f"file in the current directory or generate a new one with t"
                  f"he --train option.")
            sys.exit(1)

    if args.train:
        dataset = load_packet_dataset(*args.train)
        model.ModelTrainer(prediction_model, device, args.model_file).train(dataset)

    if args.test:
        dataset = load_packet_dataset(*args.test)
        model.ModelTester(prediction_model, device).test(dataset)

    if not args.train and not args.test:
        packet_sniffer = sniffer.PacketSniffer(prediction_model, device,
                                               args.capture_file)

        app = create_app(packet_sniffer)

        app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
