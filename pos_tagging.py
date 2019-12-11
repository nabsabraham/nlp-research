# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:14:11 2019

@author: Nabila Abraham
"""

import nltk

'''
    Part of Speech tagging assigns a tag to each part of speech (duh)
    Speech tags used in this code are : 
        (full list found at super helpful tutorial https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/)
        CD cardinal digit
        DT determiner
        IN preposition/subordinating conjunction
        JJ adjective 'big'
        JJR adjective, comparative 'bigger'
        JJS adjective, superlative 'biggest'
        NN noun, singular 'desk'
        NNS noun plural 'desks'
        NNP proper noun, singular 'Harrison'
        NNPS proper noun, plural 'Americans'
        RB adverb very, silently,
        RBR adverb, comparative better
        RBS adverb, superlative best
        RP particle give up
        TO to go 'to' the store.
        VB verb, base form take
        VBD verb, past tense took
        VBG verb, gerund/present participle taking
        VBN verb, past participle taken
        
'''
def get_tagged_words(sentence):
    sent = nltk.sent_tokenize(sentence)
    tokens = [nltk.word_tokenize(s) for s in sent]
    tagged_words = [nltk.pos_tag(t) for t in tokens]
    return tagged_words

def get_phrases(tagged_words, patterns):
    '''
    ? -> match 0 or 1 
    * -> match 0 or more
    + -> match 1 or more 
    . -> any character except new line 
    '''
    NPChunker = nltk.RegexpParser(patterns)
    phrase_tree = [NPChunker.parse(t) for t in tagged_words]
    phrases = []
    
    for tree in phrase_tree:
        for subtree in tree.subtrees():
#            print(subtree.label())
#            subtree.draw()
            if subtree.label() != 'S':
                t = subtree
                t = ' '.join(word for word, tag in t.leaves())
                phrases.append(t)
    return phrases

patterns = """
            NP: {<NN.*><IN*>?<JJ*>?<NN.+>+}     
            NP: {<JJ?|DT?><NN.*>+}
            VP: {<VB.?>*<RB?><NP|PP>*}
            VP: {<VB.?><IN?|TO?><CD?><IN?|TO?><CD?>}
            """

sentence = "I like big butts and I cannot lie" 
tagged = get_tagged_words(sentence)
phrase = get_phrases(tagged, patterns)


