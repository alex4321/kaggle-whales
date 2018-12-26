from ignite.engine import _prepare_batch


class ChainThawOptimizerStep:
    def __init__(self, model, steps, optimizer_builder, criterion, device):
        self.model = model
        self.steps = steps
        self.optimizer_builder = optimizer_builder
        self.device = device
        self.criterion = criterion
        # State
        self.current_epoch = None
        self.current_optimizer = None

    def epoch_preparation(self, epoch_idx):
        trainable_param_names, optimizer_params = self.steps[epoch_idx % len(self.steps)]
        trainable_params = []
        for name, param in self.model.named_parameters():
            trainable = name in trainable_param_names
            param.requires_grad = trainable
            if trainable:
                trainable_params.append(param)
        self.current_optimizer = self.optimizer_builder(trainable_params, **optimizer_params)

    def step(self, engine, batch):
        if engine.state.epoch != self.current_epoch:
            self.current_epoch = engine.state.epoch
            self.epoch_preparation(self.current_epoch - 1)
        self.model.train()
        self.current_optimizer.zero_grad()
        x, y_true = _prepare_batch(batch, device=self.device)
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y_true)
        loss.backward()
        self.current_optimizer.step()
        #return loss.item()
        return y_pred, y_true

    def __call__(self, engine, batch):
        return self.step(engine, batch)
