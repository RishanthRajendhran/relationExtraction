import nltk

#FunctionName: 
#   matchWordInSentence
#Input:
#   sentences       :   A sentence (str)
#   word            :   Word to match for
#   debugMode       :   Boolean variable to enable debug mode
#                        Default: False
#Output:
#   bestMatch    :  Closest match for word in sentence
#   bestRecall   :  Recall of the closest match wrt the word
#Description:
#   This function is used to find the closest match for a word 
#   in a given sentence in terms of recall
#Notes:
#   None
def matchWordInSentence(sentence, word, debugMode=False):
    wordsInSentence = nltk.tokenize.word_tokenize(sentence)
    wordsInWord = nltk.tokenize.word_tokenize(word)
    bestRecall, bestMatch = 0, ""
    for w in range(len(wordsInWord)):
        word = wordsInWord[w]
        if word in sentence:
            indices = [i for i, x in enumerate(wordsInSentence) if x == word]
            for i in indices:  
                score = 1 
                pos = i+1
                wInd = w+1 
                matchedPhrase = [word]
                while pos < len(wordsInSentence) and wInd < len(wordsInWord):
                    if wordsInWord[wInd] ==  wordsInSentence[pos]:
                        matchedPhrase.append(wordsInWord[wInd])
                        pos += 1
                        wInd += 1
                        score += 1 
                    else: 
                        wInd += 1
                recall = score/len(wordsInWord)
                matchedPhrase = " ".join(matchedPhrase)
                if recall >= bestRecall:
                    bestRecall = recall 
                    bestMatch = matchedPhrase
    return bestMatch, bestRecall
            