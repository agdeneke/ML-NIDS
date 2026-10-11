import sniffer
import pandas as pd
from flask import Flask
from flask_cors import CORS


def create_app(packet_sniffer: sniffer.PacketSniffer):
    app = Flask(__name__)
    CORS(app)

    @app.route("/api/attacks")
    def attack_packets():
        captured_packets_df = pd.DataFrame(packet_sniffer.snapshot(), columns=["Time", "Source", "Length", "Is_ARP", "Is_TCP", "ARP_request_rate", "TCP_rate", "Is_attack"])

        return captured_packets_df[captured_packets_df["Is_attack"] == 1].to_json(orient="records")

    return app
