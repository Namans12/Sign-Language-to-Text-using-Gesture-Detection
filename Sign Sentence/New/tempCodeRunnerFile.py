import nltk
from nltk.corpus import wordnet as wn
import re
from nm_pipeline import NMPipeline
from k2t_pipeline import K2TPipeline
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from keytotext import trainer, make_dataset

train_df = make_dataset('common_gen', split='train')
test_df = make_dataset('common_gen', split='test')

model = trainer()
model.from_pretrained(model_name="t5-small")
model.train(train_df=train_df, test_df=test_df,
            batch_size=4, max_epochs=3, use_gpu=True)
model.upload(hf_username="bhuvan", model_name="k2t-test3")


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


def nlp(sentence):
    nlp2 = pipeline("mrm8488/t5-base-finetuned-common_gen")
    return nlp2(sentence)