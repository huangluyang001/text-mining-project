import nltk

from nltk.corpus import gutenberg

#books = nltk.corpus.gutenberg.fileids()
book_word = gutenberg.words()
#book_word = [word.lower() for word in book_word]
book_sent = gutenberg.sents()
#book_sent = [[word.lower() for word in sent] for sent in book_sent]

class toy_LM():
    def __init__(self):
        self.start_token = '<s>'
        self.end_token = '</s>'
        self.count_dict = {}

    def unigram_count(self, book_in_word):
        word_number = len(book_in_word)
        unigram_dict = {}
        for word in book_in_word:
            try:
                unigram_dict[word] += 1
            except KeyError:
                unigram_dict.setdefault(word, 1)
        return unigram_dict, word_number

    def ngram_count(self, book_in_sent, n=2):
        ngram_dict = {}
        sentence_number = len(book_in_sent)
        for sentence in book_in_sent:
            sentence.insert(0,self.start_token)
            sentence.append(self.end_token)
            sentence_length = len(sentence)
            for i in range(sentence_length-n+1):
                token_list = []
                for j in range(n):
                    token_list.append(sentence[i+j])
                token = ' '.join(token_list)
                try:
                    ngram_dict[token] += 1
                except KeyError:
                    ngram_dict.setdefault(token, 1)
        return ngram_dict, sentence_number


    def unsmoothed_bigram(self, unigram_dict, ngram_dict, sentence_number):
        ngram_prob_dict = {}
        for token, frequency in ngram_dict.items():
            word = token.split(' ')[0]
            if word == self.start_token:
                total = sentence_number
            else:
                try:
                    total = unigram_dict[word]
                except KeyError:
                    print('KeyError1',word)
                    continue
            prob = int(frequency)/ total
            ngram_prob_dict.setdefault(token, prob)
        return ngram_prob_dict

    def unmoothed_trigram(self, bigram_dict, trigram_dict): # Also can be used in ngram, but trigram usually works well
        trigram_prob_dict = {}
        for token, frequency in trigram_dict.items():
            word_list = token.split(' ')
            word_list.pop()
            word = ' '.join(word_list)
            try:
                total = bigram_dict[word]
            except KeyError:
                print('KeyError2', word)
                continue
            prob = int(frequency)/ total
            trigram_prob_dict.setdefault(token, prob)
        return trigram_prob_dict

    def unsmoothed_unigram(self, unigram_dict, word_number):
        unigram_prob_dict = {}
        for word, value in unigram_dict.items():
            value = value / word_number
            unigram_prob_dict.setdefault(word, value)
        return unigram_prob_dict

    def add1_smoothed_unigram(self, unigram_dict, word_number):
        word_set = set()
        unigram_prob_dict = {}
        with open('vocab.txt',encoding='utf-8') as f:
            for line in f:
                word_set.add(line.strip())
        for word, value in unigram_dict.items():
            word_set.add(word)
        vocab_number = len(word_set)
        for word in word_set:
            try:
                value = (unigram_dict[word]+1) / (word_number+vocab_number)
            except KeyError:
                value = 1/ (word_number+vocab_number)
            unigram_prob_dict.setdefault(word, value)
        return  unigram_prob_dict

    def good_turing_method(self, count_dict):
        for word, count in count_dict.items():
            try:
                self.count_dict[count] += 1
            except:
                self.count_dict.setdefault(count, 1)

    def good_turing_unigram(self, word_number, unigram_dict):
        word_set = set()
        unigram_prob_dict = {}
        self.good_turing_method(count_dict=unigram_dict)
        vocab_number_1 = len(unigram_dict)
        with open('vocab.txt', encoding='utf-8') as f:
            for line in f:
                word_set.add(line.strip())
        for word, value in unigram_dict.items():
            word_set.add(word)
        vocab_number = len(word_set)
        self.count_dict.setdefault(0, vocab_number-vocab_number_1)
        print(vocab_number-vocab_number_1)
        for word in word_set:
            try:
                count = unigram_dict[word]
            except KeyError:
                count = 0
            if count <= 10:
                value = (count + 1) * self.count_dict[count+1]/(self.count_dict[count] * word_number)
            else:
                value = count / word_number
            unigram_prob_dict.setdefault(word, value)
        return  unigram_prob_dict

    def add1_smoothed_bigram(self, unigram_dict, ngram_dict, sentence_number):
        ngram_prob_dict = {}
        word_set = set()
        vocab_size = len(ngram_dict)
        for token, frequency in ngram_dict.items():
            word = token.split(' ')[0]
            if word == self.start_token:
                total = sentence_number
            else:
                try:
                    total = unigram_dict[word]
                except KeyError:
                    print('KeyError1',word)
                    continue
            prob = (int(frequency)+1)/ (total+vocab_size)
            ngram_prob_dict.setdefault(token, prob)
        return ngram_prob_dict

    def add1_smoothed_trigram(self, bigram_dict, trigram_dict): # Also can be used in ngram, but trigram usually works well
        trigram_prob_dict = {}
        vocab_size = len(trigram_dict)
        for token, frequency in trigram_dict.items():
            word_list = token.split(' ')
            word_list.pop()
            word = ' '.join(word_list)
            try:
                total = bigram_dict[word]
            except KeyError:
                print('KeyError2', word)
                continue
            prob = (int(frequency)+1)/ (total+vocab_size)
            trigram_prob_dict.setdefault(token, prob)
        return trigram_prob_dict





'''
test
'''
if __name__ == '__main__':
    model =  toy_LM()
    unigram_dict, word_number = model.unigram_count(book_in_word=book_word)
    bigram_dict, sentence_number = model.ngram_count(book_in_sent=book_sent,
                                                     n=2)
    trigram_dict, sentence_number = model.ngram_count(book_in_sent=book_sent,
                                                      n=3)
    bigram_prob_dict = model.unsmoothed_bigram(unigram_dict=unigram_dict,
                                               ngram_dict=bigram_dict,
                                               sentence_number=sentence_number)
    trigram_prob_dict = model.unmoothed_trigram(bigram_dict=bigram_dict,
                                                trigram_dict=trigram_dict)
    unigram_prob_dict = model.unsmoothed_unigram(unigram_dict=unigram_dict,
                                                 word_number=word_number)
    with open('unigram.txt',mode='w',encoding='utf-8') as f:
        for token, value in unigram_prob_dict.items():
            thisline = token + '\t' + str(value) + '\n'
            f.write(thisline)






