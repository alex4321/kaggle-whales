import numpy as np
from ignite.engine import Engine, _prepare_batch
import torch


class Predictor:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, dataloader):
        batch_predictions = []

        def _batch_predict(_, batch):
            X, _ = _prepare_batch(batch, device=self.device)
            self.model.eval()
            with torch.no_grad():
                batch_predictions.append(self.model(X).cpu().detach().numpy())

        Engine(_batch_predict).run(dataloader, max_epochs=1)
        y_pred = np.vstack(batch_predictions)
        return y_pred
