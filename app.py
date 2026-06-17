import sniffer
import pandas as pd
from flask import Flask
from flask_cors import CORS


def create_app(packet_sniffer: sniffer.PacketSniffer):
    app = Flask(__name__)
    CORS(app)

    @app.route("/api/attacks")
    def attack_packets():
        return pd.DataFrame(packet_sniffer.captured_packets).to_json()

    return app
