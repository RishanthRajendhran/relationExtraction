import logging
import spacy 

#FunctionName: 
#   extractNERtags
#Input:
#   sentence    :   Sentence (str) 
#   debugMode   :   [Deprecated] Boolean variable to enable debug mode
#                   Default: False
#Output:
#   ner    :   List of (name, ner type) tuples 
#Description:
#   This function is used to find named entities in a sentence
#Notes:
#   None
def extractNERtags(sentence, debugMode=False):
    sentence = sentence.replace("\n","")
    nlp = spacy.load("en_core_web_lg", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
    # nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    res = nlp(sentence)
    ner = [(w.text, w.label_) for w in res.ents]
    logging.debug(f"{len(ner)} named entities extracted from sentence!")
    return ner