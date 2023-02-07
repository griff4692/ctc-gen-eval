import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data_utils.constructed_discriminative_dataset import ConstructedDiscriminativeDataset

from data_utils.data_utils import get_dataloaders
from models.discriminative_aligner import DiscriminativeAligner


BATCH_SIZE = 16
ACCUMULATE_GRAD_BATCHES = 1
NUM_WORKERS = 8
WARMUP_PROPORTION = 0.1
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.01
LR = 1e-5
VAL_CHECK_INTERVAL = 0.25


def main(args):
    splits = ['train', 'validation']
    dataset = {split: ConstructedDiscriminativeDataset(split, args.algorithm, args.data_dir) for split in splits}

    dataloader = get_dataloaders(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn='raw')

    model = DiscriminativeAligner(aggr_type=None, use_bio_lm=args.use_bio_lm)

    train_steps = args.max_epochs * (
            len(dataloader['train']) // ACCUMULATE_GRAD_BATCHES + 1)
    warmup_steps = int(train_steps * WARMUP_PROPORTION)
    model.set_hparams(
        batch_size=BATCH_SIZE,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        lr=LR,
        train_steps=train_steps,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        adam_epsilon=ADAM_EPSILON)

    suffix = '' if args.use_bio_lm else '_non_bio'
    experiment_dir = os.path.join(args.data_dir, 'weights', args.algorithm + suffix)
    print(f'Saving weights in {experiment_dir}...')
    os.makedirs(experiment_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        mode='max',
        save_top_k=1,
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=checkpoint_callback,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        val_check_interval=VAL_CHECK_INTERVAL,
        gpus=[args.device],
        default_root_dir=experiment_dir,
    )

    trainer.fit(model, dataloader['train'], dataloader['validation'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parser to Fine-Tune Discriminative CTC Model')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--algorithm', required=True)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--data_dir', default='/nlp/projects/summarization/kabupra/cumc/ctc')
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('-use_bio_lm', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
