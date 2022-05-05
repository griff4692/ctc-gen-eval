import regex as re
import random
import spacy
import torch

from nltk.corpus import stopwords
from benepar import BeneparComponent

from transformers import BartConfig, BartForConditionalGeneration, BartTokenizerFast


MAX_LENGTH = 256
SAMPLING_TOPP = 0.95
en_stopwords = stopwords.words('english')
re_special_tokens = [
    '.', '^', '$', '*', '+', '?', '|', '(', ')', '{', '}', '[', ']']


CLINBART_WEIGHTS = '/nlp/projects/kabupra/cumc/clinbart/weights/baseline/clinbart/34uqv28u/checkpoints/' \
                   'epoch=9-step=477919.ckpt'

HF_MODEL = 'facebook/bart-base'


class HallucinationGenerator:
    def __init__(self, device):
        self._device = device
        self._nlp_tokenizer = spacy.load('en_core_web_sm')

        print('Loading self-attentive parser from Benepar')
        self._parser = spacy.load('en_core_web_sm')
        self._parser.add_pipe('benepar', config={'model': 'benepar_en3_large'})

        bart_config = BartConfig.from_pretrained(HF_MODEL)
        self._tokenizer = BartTokenizerFast.from_pretrained(HF_MODEL)
        print(f'Loading clinical BART weights from {CLINBART_WEIGHTS}')
        weights = torch.load(CLINBART_WEIGHTS, map_location=f'cuda:{device}')['state_dict']
        just_bart = {k.replace('model.', '', 1): v for k, v in weights.items() if k.startswith('model.')}
        self._infiller = BartForConditionalGeneration(config=bart_config).to(self._device)
        self._infiller.load_state_dict(state_dict=just_bart)

    def parse(self, text):
        return list(self._parser(text).sents)

    def depth_first_search(self, root, sent_length, depth=0):
        children = list(root._.children)

        template = '' if len(children) != 0 else root.text
        layer = 0
        answers = []
        for child in children:
            child_result = self.depth_first_search(
                root=child, sent_length=sent_length, depth=depth+1)

            template = template + ' ' + child_result['template']
            layer = max(layer, child_result['layer'] + 1)
            answers.extend(child_result['answers'])

        if root.text.lower() in en_stopwords or \
                not any([ch.isalnum() for ch in root.text]):
            p_mask = 0.
        else:
            p_mask = len(root.text.split()) / sent_length / (layer + 1)

        if random.random() < p_mask:
            template = '<mask>'
            answers = [root.text]

        return {
            'template': template.strip(),
            'layer': layer,
            'answers': answers
        }

    def hallucinate_sent(self, root):
        for _ in range(5):
            result = self.depth_first_search(
                root=root, sent_length=len(root.text.split()))

            if result['template'] != '<mask>' and \
                    '<mask> <mask>' not in result['template'] and \
                    len(result['answers']) > 0:
                break
            else:
                result = None

        if result is None:
            return None

        # Fix template to look like original sentence
        resolved_template = str(root)
        for answer in result['answers']:
            resolved_template = resolved_template.replace(answer, '<mask>')

        result['template_ctc'] = result['template']
        result['template'] = resolved_template

        input_ids = self._tokenizer(
            [result['template']], return_tensors='pt', max_length=512, truncation=True
        ).to(self._device)['input_ids']

        gen_text = self._infiller.generate(
            input_ids=input_ids,
            max_length=MAX_LENGTH,
            top_p=0.95,
            do_sample=True
        )

        gen_text = self._tokenizer.batch_decode(gen_text, skip_special_tokens=True)[0]
        pattern = result['template']
        for special_token in re_special_tokens:
            pattern = pattern.replace(special_token, f'\{special_token}', 1)
        pattern = re.sub(r'\s?<mask>\s?', '(.*)', pattern)
        pattern = pattern + '$'

        try:
            matching = re.match(pattern=pattern, string=gen_text, flags=re.I)
        except:
            matching = None

        if matching is None:
            return None
        else:
            fillings = list(matching.groups())
            result['original_text'] = self.cleantext(root.text)
            result['gen_text'] = gen_text
            result['fillings'] = fillings

            return result

    def cleantext(self, text):
        return ' '.join([token.text for token in self._nlp_tokenizer(text)])

    def hallucinate(self, input_text):
        roots = self.parse(input_text)

        result = {key: [] for key in [
            'template', 'original_text', 'gen_text', 'answers', 'fillings']}
        for root in roots:
            sent_result = self.hallucinate_sent(root=root)
            if sent_result is None:
                return None

            for key in result:
                if key in ['answers', 'fillings']:
                    result[key].extend(sent_result[key])
                else:
                    result[key].append(sent_result[key])

        for key in ['template', 'original_text', 'gen_text']:
            result[key] = ' '.join(result[key])

        return result
