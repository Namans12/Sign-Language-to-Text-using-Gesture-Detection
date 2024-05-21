labels = ["I", "how", "go", "walk", "what", "father",
          "name", "love", "eat", "beautiful", "stop", "you", "hi"]

dict = {
    "I eat ": "I want to eat",
    "how you ": "How are you",
    "I go walk ": "I go to walk",
    "I walk ": "I go to walk",
    "what you name ": "What is your name",
    "what you father name ": "What is your father name",
    "I love father ": "I love my father",
    "you beautiful ": "You are looking beautiful"
}


def nlp(sentence):
    return dict.get(sentence, "I don't understand.")