import logging
import spacy 
import en_core_web_sm
from Modules.helper.functions.features.extractNERtags import extractNERtags

#FunctionName: 
#   maskWordWithNER
#Input:
#   sentence        :   Sentence (str) 
#   word            :   Word to mask in sentence
#   NERdefault      :   Default NER tag in case NER fails to recognize word
#   subjObjLabel    :   Label to indicate whether word is the subject or the object
#                       Choices: ["S", "O"]
#   debugMode       :   Boolean variable to enable debug mode
#                       Default: False
#Output:
#   sentence        :   Modified sentence with word masked with NER type
#Description:
#   This function is used to replace word mention with NER type
#Notes:
#   None
def maskWordWithNER(sentence, word, NERdefault, subjObjLabel, debugMode=False):
    if subjObjLabel not in ["S", "O"]:
        logging.error(f"Invalid subject/object label: {subjObjLabel}!")
        return None
    if word not in sentence:
        logging.error(f"{word} not present in {sentence}!")
        return None
    #Disabling  Spacy NER temporarily for Phase 2
    # ner = extractNERtags(sentence, debugMode)
    NERtype = None
    # for (w, t) in ner:
    #     if w == word: 
    #         NERtype = t 
    #         break 
    if NERtype == None:
        NERtype = NERdefault
    wInd = sentence.index(word)
    sentence = sentence[:wInd] + "<" + NERtype + "-" + subjObjLabel + ">" + word + "</" + NERtype + "-" + subjObjLabel + ">" + sentence[wInd+(len(word)):]
    return sentence