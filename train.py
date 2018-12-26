from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import RunningAverage, Loss
from ignite.engine import Events, Engine, create_supervised_evaluator
import torch
import common
from chain_thaw_optimizer_step import ChainThawOptimizerStep
import datasets
import models
from triplet_loss import TripletLoss
import train_config
from triplet_distance_metric import TripletDistanceMetric
import time


def _build_model():
    resnet = models.embedding_resnet_from_pretrained(common.VECTOR_SIZE)
    normed_resnet = models.L2NormedModel(resnet)
    triplet = models.TripletNetwork(normed_resnet)
    return triplet


def build_val_evaluator(model, val_dataset, metrics):
    val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=common.DEVICE)

    def _run(engine):
        val_evaluator.run(val_dataset)
        engine.state.metrics = dict(engine.state.metrics,
                                    **val_evaluator.state.metrics)

    return _run


def build_metrics_visualizer(patterns):
    def _run(engine):
        print('')
        for pattern in patterns:
            print(pattern.format(epoch=engine.state.epoch,
                                 **engine.state.metrics))
        print('')

    return _run


if __name__ == '__main__':
    train_mapping, val_mapping, _ = datasets.get_mappings(common.TRAIN_MAPPING)
    train_triplet = datasets.triplet_df_by_mapping(train_mapping)
    val_triplet = datasets.triplet_df_by_mapping(val_mapping)

    images = common.pickle_read(common.TRAIN_IMAGES_PICKLE_PATH)

    train_triplet = train_triplet
    val_triplet = val_triplet

    train_triplet_dataset = datasets.TripletDataset(images, train_triplet)
    val_triplet_dataset = datasets.TripletDataset(images, val_triplet)

    train_triplet_loader = train_triplet_dataset.build_dataloader(common.TRIPLET_BATCH_SIZE)
    val_triplet_loader = val_triplet_dataset.build_dataloader(common.TRIPLET_BATCH_SIZE)

    model = _build_model().to(common.DEVICE)
    loss = TripletLoss(common.VECTOR_SIZE, 10.0)
    optimizer_step = ChainThawOptimizerStep(
        model,
        train_config.CHAIN_THAW_STEPS,
        torch.optim.SGD,
        loss,
        common.DEVICE
    )

    trainer = Engine(optimizer_step)
    train_loss_metric = RunningAverage(Loss(loss,
                                            batch_size=lambda X: common.TRIPLET_BATCH_SIZE),
                                       0.5)
    train_loss_metric.attach(trainer, 'running_avg_loss')
    progressbar = ProgressBar(persist=True)
    progressbar.attach(trainer, metric_names=['running_avg_loss'])
    val_metrics = {
        'val_loss': Loss(loss, batch_size=lambda X: common.TRIPLET_BATCH_SIZE),
        'val_positive_distances': TripletDistanceMetric(common.VECTOR_SIZE,
                                                        2 * common.VECTOR_SIZE),
        'val_negative_distances': TripletDistanceMetric(2 * common.VECTOR_SIZE,
                                                        3 * common.VECTOR_SIZE)
    }
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              build_val_evaluator(model, val_triplet_loader, val_metrics))
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              build_metrics_visualizer([
                                  'Epoch {epoch}',
                                  'Validation loss: {val_loss}',
                                  'Anchor-positive distances: {val_positive_distances}',
                                  'Anchor-negative distances: {val_negative_distances}',
                              ]))

    models_directory = common.TRIPLET_MODELS_DIRECTORY.format(time=int(time.time()))
    checkpoint = ModelCheckpoint(dirname=models_directory,
                                 filename_prefix='triplet-model',
                                 score_function=lambda engine: -engine.state.metrics['val_loss'],
                                 require_empty=True,
                                 create_dir=True)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint, {
        'model': model
    })

    early_stopping = EarlyStopping(patience=common.TRIPLET_EARLY_STOPPING_PATIENCE,
                                   score_function=lambda engine: -engine.state.metrics['val_loss'],
                                   trainer=trainer)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, early_stopping)

    trainer.run(train_triplet_loader, max_epochs=common.TRIPLET_MAX_EPOCHS)
