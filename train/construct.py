import os
import ujson
import regex as re
from tqdm import tqdm
import numpy as np
import argparse
from summa import summarizer

from models.hallucination_generator import HallucinationGenerator


HTML_REGEX_NO_SPACE = r'(<[a-z][^>]+>|<\/?[a-z]>)'
def remove_tags_from_sent(str):
    return re.sub(HTML_REGEX_NO_SPACE, '', str)


def construct_summ(example, para_generator, hallu_generator):
    n_summ_sents = np.random.randint(1, 4)

    summ_sents = None
    for ratio in np.arange(0.1, 1., 0.1):
        if len(example['src'].split()) > 1000:
            example['src'] = ' '.join(example['src'].split()[:1000])
        summ_sents = summarizer.summarize(
            example['src'], ratio=ratio).split('\n')

        if len(summ_sents) >= n_summ_sents:
            sent_idxes = np.random.choice(
                len(summ_sents), n_summ_sents, replace=False).tolist()
            sent_idxes.sort()
            summ_sents = [summ_sents[idx] for idx in sent_idxes]

    if summ_sents is None:
        return None
    else:
        tgt = ' '.join(summ_sents)
        para_tgt = para_generator.generate(input_text=tgt)

    if para_tgt is None:
        return None

    hallu = hallu_generator.hallucinate(input_text=para_tgt)

    if hallu is None:
        return None

    return {
        'src': example['src'],
        'ref': example['ref'],
        'tgt': tgt,
        'para_tgt': hallu['original_text'],
        'template': hallu['template'],
        'hallu_tgt': hallu['gen_text'],
        'answers': hallu['answers'],
        'fillings': hallu['fillings']
    }


def construct_summ_ref(input_text, hallu_generator):
    hallu = hallu_generator.hallucinate(input_text=input_text)
    
    if hallu is None: 
        return None
    
    return {
        'para_tgt': hallu['original_text'],
        'template': hallu['template'],
        'hallu_tgt': hallu['gen_text'],
        'answers': hallu['answers'],
        'fillings': hallu['fillings']
    }


def main(args):
    print(f'Reading in data from {args.data_fn}')

    with open(args.data_fn, 'r') as fd:
        examples = [ujson.loads(x.strip()) for x in fd.readlines() if len(x.strip()) > 0]

    # Randomly shuffle the data
    np.random.seed(1992)
    np.random.shuffle(examples)

    hallu_generator = HallucinationGenerator(device=args.device)
    results = []
    for example in tqdm(examples, desc=f'Constructing'):
        target = remove_tags_from_sent(example['summary'])
        if len(target.split(' ')) > args.max_sent_toks:
            continue
        example['context'] = remove_tags_from_sent(example.pop('text'))

        halluc_info = construct_summ_ref(
            input_text=target,
            hallu_generator=hallu_generator
        )
        if halluc_info is None:
            continue
        example.update(halluc_info)

        if example is not None:
            results.append(example)
            if len(results) % 100 == 0:
                print(f'{len(results)} examples constructed.')

            if len(results) >= args.target_size:
                break
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parser to generate BART Mask-and-Fill')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--data_dir', default='/nlp/projects/summarization/kabupra/cumc')
    parser.add_argument('--max_sent_toks', default=128, type=int)
    parser.add_argument('--algorithm', default='gain_rouge')
    parser.add_argument('--target_size', type=int, default=10000)
    parser.add_argument('--splits', type=str, default='validation,train')
    args = parser.parse_args()

    for split in args.splits.split(','):
        args.data_fn = os.path.join(args.data_dir, 'human', 'bart_align_datasets', f'{split}_{args.algorithm}.json')
        args.out_dir = os.path.join(args.data_dir, 'ctc')
        os.makedirs(args.out_dir, exist_ok=True)
        args.out_fn = os.path.join(args.out_dir, f'{split}_{args.algorithm}.json')

        results = main(args)
        print(f'Saving {len(results)} results to {args.out_fn}')
        with open(args.out_fn, 'w') as fd:
            ujson.dump(results, fd)
