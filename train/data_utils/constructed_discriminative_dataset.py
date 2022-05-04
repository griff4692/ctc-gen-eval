import os
import ujson

from torch.utils.data import Dataset

from data_utils.data_utils import text_clean, get_discriminative_token_labels, get_context, TokenClassificationExample


class ConstructedDiscriminativeDataset(Dataset):
    def __init__(self, split, data_dir):
        self._examples = []

        data_fn = os.path.join(data_dir, f'{split}.json')
        print(f'Loading data from {data_fn}')
        with open(data_fn, 'r') as fd:
            examples = ujson.load(fd)

        for example in examples:
            input_text, labels = get_discriminative_token_labels(
                template=example['template'],
                answers=example['answers'],
                fillings=example['fillings'])

            context = text_clean(' '.join(example['context_sents']))
            input_text = text_clean(input_text)
            example = TokenClassificationExample(context=context, input_text=input_text, labels=labels)
            self._examples.append(example)

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):
        return self._examples[item]
