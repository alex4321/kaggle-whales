import math
import torch


class TripletLoss(torch.nn.Module):
    def __init__(self, vector_size, distance_multiplier):
        super(TripletLoss, self).__init__()
        self.vector_size = vector_size
        self.distance_multiplier = distance_multiplier
        self.prevrun = {}

    def _difference(self, y1, y2):
        diff = ((y1 - y2) ** 2).sum(dim=-1)
        # Avoid https://github.com/pytorch/pytorch/issues/4320
        #  NaN occures because of next things:
        #  - std(...) calculated as sqrt(mean((x - mean(x))**2))
        #  - sqrt gradient not defined at zero
        #  - so when mean((x-mean(x))**2) goes zero - gradient goes NaN
        #  - it's occured for me when I tryed to overfit model for test
        #  - so than I added component that's disturb such "ideal" solution
        # TODO: But in better case I must implement both forward and bacward steps
        diff[-1:] += 1e-8
        return diff

    def forward(self, y_pred, y_true):
        y_anchor = math.sqrt(self.distance_multiplier) * y_pred[:, :self.vector_size]
        y_positive = math.sqrt(self.distance_multiplier) * y_pred[:, self.vector_size:2*self.vector_size]
        y_negative = math.sqrt(self.distance_multiplier) * y_pred[:, 2*self.vector_size:]
        positive_difference = self._difference(y_anchor, y_positive)
        negative_difference = self._difference(y_anchor, y_negative)
        positive_loss_elementwise = positive_difference + torch.std(positive_difference)
        negative_loss_elementwise = -negative_difference + torch.std(negative_difference) - (
            self.distance_multiplier ** 2 - torch.abs(negative_difference.mean())
        )
        loss_elementwise = positive_loss_elementwise + negative_loss_elementwise
        loss = loss_elementwise.mean()
        return loss
