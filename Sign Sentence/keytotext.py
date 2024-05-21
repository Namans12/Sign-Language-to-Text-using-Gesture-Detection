# keytotext.py
from keytotext_module import NMPipeline, K2TPipeline, nltk, words_to_sentence, nlp

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
   
