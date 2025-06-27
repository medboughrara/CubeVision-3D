import time
import csv
import os
import paho.mqtt.client as mqtt

# Map class_id to color
CLASS_ID_TO_COLOR = {
    "2": "red",
    "0": "blue",
    "1": "green"
}

class CubeMQTTPublisher:
    def __init__(self, csv_path, mqtt_host="localhost", mqtt_port=1883, topic="cube/detection"):
        self.csv_path = csv_path
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.topic = topic
        self.last_class_id = None
        self.last_line = 0
        self.client = mqtt.Client()
        self.client.connect(self.mqtt_host, self.mqtt_port, 60)
        self.client.loop_start()

    def monitor_csv(self):
        while True:
            if not os.path.exists(self.csv_path):
                time.sleep(1)
                continue

            with open(self.csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if len(rows) > self.last_line:
                    for row in rows[self.last_line:]:
                        class_id = str(row["class_id"])
                        x = row["x"]
                        y = row["y"]
                        if class_id in CLASS_ID_TO_COLOR:
                            if class_id != self.last_class_id:
                                color = CLASS_ID_TO_COLOR[class_id]
                                payload = {
                                    "color": color,
                                    "x": x,
                                    "y": y
                                }
                                # Send as a simple string, or use json.dumps(payload) for JSON
                                self.client.publish(self.topic, f"{color},{x},{y}")
                                print(f"Published: {color},{x},{y}")
                                self.last_class_id = class_id
                    self.last_line = len(rows)
            time.sleep(0.5)  # Check for new lines every 0.5s

if __name__ == "__main__":
    csv_path = "detection_history.csv"
    publisher = CubeMQTTPublisher(csv_path)
    publisher.monitor_csv()