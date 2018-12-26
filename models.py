import torch
from torchvision.models import resnet50


def embedding_resnet_from_pretrained(vector_size):
    base_model = resnet50(pretrained=True)
    state = base_model.state_dict()
    del state['fc.weight']
    del state['fc.bias']
    new_model = resnet50(pretrained=False, num_classes=vector_size)
    new_model_state = new_model.state_dict()
    state['fc.weight'] = new_model_state['fc.weight']
    state['fc.bias'] = new_model_state['fc.bias']
    new_model.load_state_dict(state)
    return new_model


class L2NormedModel(torch.nn.Module):
    def __init__(self, base):
        super(L2NormedModel, self).__init__()
        self.base = base

    def _l2_normed(self, vectors):
        l2_norms = (vectors ** 2).sum(dim=-1).sqrt().reshape([-1, 1])
        l2_normed = vectors / l2_norms
        return l2_normed

    def forward(self, *input):
        y = self.base(*input)
        y_normed = self._l2_normed(y)
        return y_normed


class TripletNetwork(torch.nn.Module):
    def __init__(self, base):
        super(TripletNetwork, self).__init__()
        self.base = base

    def forward(self, X):
        anchor, positive, negative = X
        anchor_embedding = self.base(anchor)
        positive_embedding = self.base(positive)
        negative_embedding = self.base(negative)
        output = torch.cat((anchor_embedding, positive_embedding, negative_embedding), dim=-1)
        return output