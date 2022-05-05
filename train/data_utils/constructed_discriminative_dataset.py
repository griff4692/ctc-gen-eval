import os
import regex as re
import ujson

from torch.utils.data import Dataset

from data_utils.data_utils import text_clean, get_discriminative_token_labels, get_context, TokenClassificationExample


HTML_REGEX_NO_SPACE = r'(<[a-z][^>]+>|<\/?[a-z]>)'
def remove_tags_from_sent(str):
    return re.sub(HTML_REGEX_NO_SPACE, '', str)


class ConstructedDiscriminativeDataset(Dataset):
    def __init__(self, split, data_dir):
        self._examples = []

        data_fn = os.path.join(data_dir, f'{split}.json')
        print(f'Loading data from {data_fn}')
        with open(data_fn, 'r') as fd:
            examples = ujson.load(fd)

        for example in examples:
            num_mask = len(re.findall(r'<mask>', example['template']))
            num_answer = len(example['answers'])
            if num_mask != num_answer:
                # print(f'Mask ({num_mask}) / Answer ({num_answer}) Mismatch')
                continue
            try:
                input_text, labels = get_discriminative_token_labels(
                    template=example['template'],
                    answers=example['answers'],
                    fillings=example['fillings'])
                context = text_clean(' '.join(example['context_sents']))
                input_text = text_clean(input_text)
                example = TokenClassificationExample(context=context, input_text=input_text, labels=labels)
                self._examples.append(example)
            except Exception as e:
                print('Cannot process this example: ', e)
        print(f'{len(self._examples)}/{len(examples)} are valid...')

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, item):
        return self._examples[item]
