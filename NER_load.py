import spacy

nlp1 = spacy.load(r"./output/model-best") #load the best model


doc = nlp1("snow") # input sample text

doc.ents

print(doc.ents)

print([(ent.text, ent.label_) for ent in doc.ents])
