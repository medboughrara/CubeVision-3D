import datetime

class DetectionHistory:
    def __init__(self):
        self.history = []

    def add(self, detection):
        self.history.append(detection)

    def get_all(self):
        return self.history

    def save_csv(self, filename):
        import csv
        if not self.history:
            return
        keys = self.history[0].keys()
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)