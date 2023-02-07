import os
import json


DATA_DIR = '/nlp/projects/summarization/kabupra/cumc'
OTHER = [
    'top_k_rouge',
    'top_k_bert',
    'gain_bert',
    'top_k_section',
    'entity_chain'
]


for split in ['validation', 'train']:
    in_fn = os.path.join(DATA_DIR, 'ctc', f'{split}_gain_rouge.json')
    with open(in_fn, 'r') as fd:
        examples = json.load(fd)

    for algo in OTHER:
        out_fn = os.path.join(DATA_DIR, 'ctc', f'{split}_{algo}.json')
        algo_fn = os.path.join(
            '/nlp/projects/summarization/kabupra/cumc/human/bart_align_datasets',
            f'{split}_{algo}.json'
        )
        with open(algo_fn, 'r') as fd:
            lines = fd.readlines()
            objs = [json.loads(x) for x in lines if len(x.strip()) > 0]
            ref2con = {}
            for obj in objs:
                ref2con[obj['summary']] = obj['text']

        examples_copy = []
        for exc in examples:
            s = exc['summary']
            if s not in ref2con:
                print('Missing.')
                continue
            exc = exc.copy()
            assert 'context' in exc.keys()
            exc['context'] = ref2con[s]
            examples_copy.append(exc)

        print(f'Saving {len(examples_copy)} to {out_fn}')
        with open(out_fn, 'w') as fd:
            json.dump(examples_copy, fd)
