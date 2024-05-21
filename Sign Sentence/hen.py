import nltk
from nltk.corpus import wordnet as wn
import re
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, pipeline, trainer, make_dataset

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def words_to_sentence(words):
    tagged_words = nltk.pos_tag(words)
    missing_words = []
    for i in range(len(tagged_words)):
        word = tagged_words[i][0]
        tag = tagged_words[i][1]
        if tag.startswith('N'):
            if i == 0 or (i > 0 and tagged_words[i-1][1] != 'DT'):
                missing_words.append('the')
            if i > 0 and tagged_words[i-1][1] not in ['IN', 'TO']:
                missing_words.append('of')
            synsets = wn.synsets(word, pos=wn.NOUN)
            if synsets:
                synset = synsets[0]
                hypernyms = synset.hypernyms()
                if hypernyms:
                    hypernym = hypernyms[0].lemmas()[0].name()
                    missing_words.append(hypernym.replace('_', ' '))
            missing_words.append(word)
        elif tag.startswith('V'):
            if i == 0 or (i > 0 and tagged_words[i-1][1] not in ['TO', 'MD']):
                missing_words.append('to')
            missing_words.append(word)

    if len(tagged_words) == 2 and tagged_words[1][1].startswith('V'):
        missing_words.insert(
            1, 'is' if tagged_words[1][1].startswith('VBZ') else 'are')

    if len(tagged_words) == 3 and tagged_words[1][1].startswith('V'):
        missing_words.insert(1, 'a')

    words = missing_words
    text = ' '.join(words)

    sentences = nltk.sent_tokenize(text)
    sentence = ' '.join(sentences)
    sentence = sentence.capitalize()

    if not sentence.endswith('.'):
        sentence += '.'

    return sentence

train_df = make_dataset('common_gen', split='train')
test_df = make_dataset('common_gen', split='test')

model = trainer()
model.from_pretrained(model_name="t5-small")
model.train(train_df=train_df, test_df=test_df,
            batch_size=4, max_epochs=3, use_gpu=True)
model.upload(hf_username="bhuvan", model_name="k2t-test3")

class NMPipeline:
    def __init__(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, use_cuda: bool
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

        if self.model.__class__.__name__ not in ["T5ForConditionalGeneration"]:
            raise AssertionError

        if "T5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "t5"

        self.default_generate_kwargs = {
            "max_length": 1024,
            "num_beams": 4,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }

    def __call__(self, keywords, **kwargs):
        inputs = self._prepare_inputs_for_nm(keywords)
        result = ""
        if not kwargs:
            kwargs = self.default_generate_kwargs

        for txt in inputs:
            input_ids = self._tokenize("{} </s>".format(txt), padding=False)
            outputs = self.model.generate(input_ids.to(self.device), **kwargs)
            result += self.tokenizer.decode(outputs[0])

        result = re.sub("<pad>|</s>", "", result)
        return result.strip()

    def _prepare_inputs_for_nm(self, keywords):
        text = str(keywords)
        text = text.replace(",", " ")
        text = text.replace("'", "")
        text = text.replace("[", "")
        text = text.replace("]", "")
        texts = text.split(".")
        return texts

    def _tokenize(
        self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=1024,
    ):
        inputs = self.tokenizer.encode(
            inputs,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt",
        )
        return inputs

class K2TPipeline:
    def __init__(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, use_cuda: bool
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

        if self.model.__class__.__name__ not in ["T5ForConditionalGeneration"]:
            raise AssertionError

        if "T5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "t5"

        self.default_generate_kwargs = {
            "max_length": 1024,
            "num_beams": 4,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }

    def __call__(self, keywords, **kwargs):
        inputs = self._prepare_inputs_for_k2t(keywords)
        result = ""
        if not kwargs:
            kwargs = self.default_generate_kwargs

        for txt in inputs:
            input_ids = self._tokenize("{} </s>".format(txt), padding=False)
            outputs = self.model.generate(input_ids.to(self.device), **kwargs)
            result += self.tokenizer.decode(outputs[0])

        result = re.sub("<pad>|</s>", "", result)
        return result.strip()

    def _prepare_inputs_for_k2t(self, keywords):
        text = str(keywords)
        text = text.replace(",", "|")
        text = text.replace("'", "")
        text = text.replace("[", "")
        text = text.replace("]", "")
        texts = text.split(".")
        return texts

    def _tokenize(
        self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=1024,
    ):
        inputs = self.tokenizer.encode(
            inputs,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt",
        )
        return inputs
