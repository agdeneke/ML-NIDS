import nids
from flask import Flask

def create_app(packet_sniffer: nids.PacketSniffer):
    app = Flask(__name__)

    @app.route("/api/attacks")
    def attack_packets():
        captured_packets_df = packet_sniffer.captured_packets_df

        return captured_packets_df[captured_packets_df["is_attack"] == 1].to_json()

    return app
