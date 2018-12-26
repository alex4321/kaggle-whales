import numpy as np
from ignite.metrics import Metric


class TripletDistanceMetric(Metric):
    def __init__(self, target_index_start, target_index_end):
        super(TripletDistanceMetric, self).__init__()
        self.target_index_end = target_index_end
        self.target_index_start = target_index_start
        self.vector_size = target_index_end - target_index_start
        self.reset()

    def reset(self):
        self.anchor_predictions = []
        self.target_predictions = []

    def update(self, output):
        assert len(output) == 2
        y_pred, _ = output
        anchor_output = y_pred[:, :self.vector_size].cpu().detach().numpy()
        target_output = y_pred[:, self.target_index_start:self.target_index_end].cpu().detach().numpy()
        self.anchor_predictions.append(anchor_output)
        self.target_predictions.append(target_output)

    def _distance(self, y1, y2):
        diff = y1 - y2
        return np.sqrt( (diff ** 2).sum(axis=-1) )

    def compute(self):
        anchor_predictions = np.vstack(self.anchor_predictions)
        target_predictions = np.vstack(self.target_predictions)
        distances = self._distance(anchor_predictions, target_predictions)
        return {
            'mean': np.mean(distances),
            'std': np.std(distances)
        }