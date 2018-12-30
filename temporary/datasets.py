class TripletDataset:
    def __init__(self, mapping, random_state, size):
        self.triplet_mapping = self._generate_triplet_mapping(mapping, random_state, size)

    def _generate_triplet_mapping(self, mapping, random_state, size):
        positives =