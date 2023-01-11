import spacy
import gensim.downloader as api
from gensim.models import Word2Vec
from spacy import displacy

import concise_concepts

# import en_core_web_sm

data = {
    # "fruit": ["apple", "pear", "orange"],
    # "vegetable": ["broccoli", "spinach", "tomato"],
    # "meat": ["beef", "pork", "fish", "lamb"],
    "dew": ["droplet", "morning", "water", "moisture"],
    "fog": ["smoke", "smog", "steam", "mist"],
    "frost": ["ice", "freeze", "blight"],
    "hail": ["hailstorm", "ice", "ball"],
    "lightning": ["bolt", "lumination", "electricity"],
    "rain": ["drizzle", "rainfall", "rainstorm", "precipitation"],
    "rainbow": ["arc", "curve", "prism"],
    "rime": ["hoar", "icicle", "ice"],
    "sandstorm": ["sand", "duster", "dust"],
    "snow": ["winter", "cold", "snowflakes"],
}

text_text = """ A car is parked in the Snow. The snow is white and it is very cold outside."""

# model = api.load("glove-wiki-gigaword-300")
# dataset = api.load("text8")  # load dataset as iterable
# print(api.info("text8"))
# model = Word2Vec(dataset)  # train w2v model


# spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_lg")

nlp.add_pipe(
    "concise_concepts",
    config={
        "data": data,
        "model_path": "glove-wiki-gigaword-300",
        "ent_score": True,  # Entity Scoring section
        "verbose": True,
        # "exclude_pos": ["VERB", "AUX"],
        # "exclude_dep": ["DOBJ", "PCOMP"],
        "include_compound_words": True,
        "json_path": "./weather_patterns.json",
    },
)
doc = nlp(text)
if len(doc.ents):
    print([(ent.text, ent.label_, ent._.ent_score) for ent in doc.ents])
else:
    print("I do not know what is in there")

print(doc.ents)
print([(ent.text, ent.label_, ent._.ent_score) for ent in doc.ents])

# options = {
#     "colors": {"fruit": "darkorange", "vegetable": "limegreen", "meat": "salmon"},
#     "ents": ["fruit", "vegetable", "meat"],
# }
#
# ents = doc.ents
# for ent in ents:
#     new_label = f"{ent.label_} ({ent._.ent_score:.0%})"
#     options["colors"][new_label] = options["colors"].get(ent.label_.lower(), None)
#     options["ents"].append(new_label)
#     ent.label_ = new_label
# doc.ents = ents
#
# displacy.render(doc, style="ent", options=options)
