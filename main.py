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

    if len(sys.argv) > 1:
        if len(sys.argv) != 4:
            print("ERROR: Incorrect number of arguments. Please specify a CSV file containing a training or test dataset, along with a CSV file containing its labels.")
            sys.exit(1)

        packet_capture_filename = sys.argv[2]
        labels_filename = sys.argv[3]
        packet_df = pd.read_csv(packet_capture_filename)
        label_df = pd.read_csv(labels_filename)

        packet_dataset = nids.preprocess(packet_df, label_df)

        if sys.argv[1] == "--train":
            model.ModelTrainer(prediction_model, device).train(packet_dataset)
        elif sys.argv[1] == "--test":
            test_dataloader = torch.utils.data.DataLoader(packet_dataset, batch_size=64)
            model.ModelTester(prediction_model, device).test_loop(test_dataloader)

    else:
        nids.PacketSniffer(prediction_model, device)

if __name__ == "__main__":
    main()
