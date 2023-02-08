import logging
import spacy 
import en_core_web_sm

#FunctionName: 
#   extractNERtags
#Input:
#   sentence    :   Sentence (str) 
#   debugMode   :   Boolean variable to enable debug mode
#                   Default: False
#Output:
#   ner    :   List of (name, ner type) tuples 
#Description:
#   This function is used to find named entities in a sentence
#Notes:
#   None
def extractNERtags(sentence, debugMode=False):
    sentence = sentence.replace("\n","")
    nlp = en_core_web_sm.load()
    res = nlp(sentence)
    ner = [(w.text, w.label_) for w in res.ents]
    if debugMode:
        logging.info(f"{len(ner)} named entities eextracted from sentence!")
    return ner