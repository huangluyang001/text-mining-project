
import nltk
import Levenshtein
import math
import lm
from nltk.corpus import gutenberg, brown, reuters

class spelling_correction():
    def __init__(self):
        self.start_token = '<s>'
        self.end_token = '</s>'
        self.word_set = set()
        self.count_dict = {}

    def word_set_get(self):
        with open('vocab.txt',encoding='utf-8') as f:
            for line in f:
                self.word_set.add(line.strip())

    def data_read(self):
        num2sent = {}
        num2error = {}
        with open('testdata.txt',encoding='utf-8') as f:
            for line in f:
                data = line.strip().split('\t')
                num2sent.setdefault(data[0],data[2])
                num2error.setdefault(data[0], data[1])
        return num2error, num2sent

    def confusion_matrix(self,debug='off'):
        count_dict = {}
        prob_dict = {}
        with open('spell-errors.txt', encoding='utf-8') as f:
            for line in f:
                data = line.strip().split(': ')
                right = data[0]
                wrong = data[1].split(', ')
                for wrong_word in wrong:
                    word = wrong_word.split('*')[0]
                    try:
                        number = int(wrong_word.split('*')[1])
                    except:
                        number = 1
                    wrong_types = Levenshtein.editops(right,word)
                    right_list = list(right)
                    right_list_2 = [(right_list[i],right_list[i+1]) for i in range(len(right_list)-1)]
                    right_list_added = ['>'] + right_list
                    right_list_2_added = [('>',right_list[0])] + right_list_2
                    wrong_list = list(word)
                    wrong_list_added = ['>'] + wrong_list
                    wrong_list_2 = [(wrong_list[i], wrong_list[i+1]) for i in range(len(wrong_list)-1)]
                    wrong_list_2_added = [('>',wrong_list[0])] + wrong_list_2
                    for candidate in wrong_list_added:
                        if candidate in right_list_added:
                            token = candidate + '|' + candidate
                            try:
                                count_dict[token] += number
                            except KeyError:
                                count_dict.setdefault(token, number)
                    for candidate in wrong_list_2_added:
                        if candidate in right_list_2_added:
                            alpha = candidate[0] + candidate[1]
                            token = alpha + '|' + alpha
                            try:
                                count_dict[token] += number
                            except KeyError:
                                count_dict.setdefault(token, number)
                    for i in range(len(wrong_types)):
                        wrong_type = wrong_types[i]
                        if i+1 < len(wrong_types):
                            wrong_type_2 = wrong_types[i+1]
                            right_id_1 = int(wrong_type[1])
                            right_id_2 = int(wrong_type_2[1])
                            wrong_id_1 = int(wrong_type[2])
                            wrong_id_2 = int(wrong_type_2[2])
                            if wrong_type[0] == 'replace' and wrong_types[i + 1][0] == 'replace':
                                right_token = right_list[right_id_1] + right_list[right_id_2]
                                wrong_token = wrong_list[wrong_id_2] + wrong_list[wrong_id_1]
                            else:
                                right_token = 1
                                wrong_token = 0
                            if wrong_type[0] == 'replace' and wrong_types[i + 1][0] == 'replace' and right_id_1 + 1 == right_id_2 and right_token == wrong_token:
                                token = right_token + '|' + wrong_list[wrong_id_1] + wrong_list[wrong_id_2]
                                try:
                                    count_dict[token] += 1
                                except KeyError:
                                    count_dict.setdefault(token, 1)
                            else:
                                right_id_1 = int(wrong_type[1])
                                wrong_id_1 = int(wrong_type[2])
                                if wrong_type[0] == 'insert':
                                    if right_id_1 != 0:
                                        token = wrong_list[wrong_id_1-1] + '|' + wrong_list[wrong_id_1-1] + wrong_list[
                                        wrong_id_1]
                                    elif right_id_1 == 0:
                                        token = '>' + '|' + '>' + wrong_list[wrong_id_1]
                                    if debug == 'on':
                                        print(right_list)
                                        print(wrong_list)
                                        print(wrong_types)
                                        print(right_id_1)
                                        print(wrong_id_1)
                                        print(token)
                                elif wrong_type[0] == 'delete':
                                    if wrong_id_1 != 0:
                                        token = right_list[right_id_1-1] + right_list[right_id_1] + '|' + right_list[
                                        right_id_1-1]
                                    elif wrong_id_1 == 0:
                                        token = '>' + right_list[right_id_1] + '|' + '>'
                                elif wrong_type[0] == 'replace':
                                    token = right_list[right_id_1] + '|' + wrong_list[wrong_id_1]
                                try:
                                    count_dict[token] += number
                                except KeyError:
                                    count_dict.setdefault(token, number)
                        else:
                            right_id_1 = int(wrong_type[1])
                            wrong_id_1 = int(wrong_type[2])
                            if wrong_type[0] == 'insert':
                                if right_id_1 != 0:
                                    token = wrong_list[wrong_id_1 - 1] + '|' + wrong_list[wrong_id_1 - 1] + wrong_list[
                                        wrong_id_1]
                                elif right_id_1 == 0:
                                    token = '>' + '|' + '>' + wrong_list[wrong_id_1]
                            elif wrong_type[0] == 'delete':
                                if wrong_id_1 != 0:
                                    token = right_list[right_id_1 - 1] + right_list[right_id_1] + '|' + right_list[
                                        right_id_1 - 1]
                                elif wrong_id_1 == 0:
                                    token = '>' + right_list[right_id_1] + '|' + '>'
                            elif wrong_type[0] == 'replace':
                                token = right_list[right_id_1] + '|' + wrong_list[wrong_id_1]
                            try:
                                count_dict[token] += number
                            except KeyError:
                                count_dict.setdefault(token, number)
        normalization_value = {}
        for key, value in count_dict.items():
            words = key.split('|')
            if words[0] == words[1]:
                normalization_value.setdefault(words[0], value)
        for key, value in count_dict.items():
            words = key.split('|')
            if words[0] != words[1]:
                try:
                    value = value/normalization_value[words[0]]
                except KeyError:
                    print(words[0])
                prob_dict.setdefault(key, value)
        return count_dict, prob_dict

    def confusion_matrix_new(self):

        count_dict = {}
        prob_dict = {}
        count_right = {}
        with open('count_1edit2.txt', encoding='ansi') as f:
            for line in f:
                data = line.strip().split('\t')
                if len(data) != 2:
                    print('Error: ', data)
                    continue
                tokens = data[0].split('|')
                if len(tokens) != 2:
                    print('Error: ', token)
                    continue
                token = tokens[1] + '|' + tokens[0]
                value = int(data[1])
                count_dict.setdefault(token, value)
        book_word = brown.words()
        for word in book_word:
            word_list_1 = ['>'] + list(word)
            word_list_2 = [(word_list_1[i], word_list_1[i+1]) for i in range((len(word_list_1)-1))]
            for token in word_list_1:
                try:
                    count_right[token] += 1
                except:
                    count_right.setdefault(token, 1)
            for tokens in word_list_2:
                token = tokens[0] + tokens[1]
                try:
                    count_right[token] += 1
                except:
                    count_right.setdefault(token, 1)
        for token, count in count_dict.items():
            token_1 = token.split('|')[0]
            try:
                value = count / count_right[token_1]
            except KeyError:
                print(token)
            prob_dict.setdefault(token, value)
        return prob_dict

    def confusion_matrix_new_smoothing(self):
        count_dict = {}
        prob_dict = {}
        count_right = {}
        with open('count_1edit2.txt', encoding='ansi') as f:
            for line in f:
                data = line.strip().split('\t')
                if len(data) != 2:
                    print('Error: ', data)
                    continue
                tokens = data[0].split('|')
                if len(tokens) != 2:
                    print('Error: ', token)
                    continue
                token = tokens[1] + '|' + tokens[0]
                value = int(data[1])
                count_dict.setdefault(token, value)
        book_word = brown.words()
        alphas = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ>'
        token_list = [alphas[i]+'|'+alphas[j] for i in range(len(alphas)) for j in range(len(alphas))] + [alphas[i]+'|'+alphas[i]+alphas[j] for i in range(len(alphas)) for j in range(len(alphas))] + [alphas[i]+alphas[j]+'|'+alphas[i] for i in range(len(alphas)) for j in range(len(alphas))] + [alphas[i]+alphas[j]+'|'+alphas[j]+alphas[i] for i in range(len(alphas)) for j in range(len(alphas))]
        for alpha in token_list:
            count_dict.setdefault(alpha, 0)
        for word in book_word:
            word_list_1 = ['>'] + list(word)
            word_list_2 = [(word_list_1[i], word_list_1[i + 1]) for i in range((len(word_list_1) - 1))]
            for token in word_list_1:
                try:
                    count_right[token] += 1
                except:
                    count_right.setdefault(token, 1)
            for tokens in word_list_2:
                token = tokens[0] + tokens[1]
                try:
                    count_right[token] += 1
                except:
                    count_right.setdefault(token, 1)
        vocab_number = len(count_dict)
        for token, count in count_dict.items():
            token_1 = token.split('|')[0]
            try:
                value = (count+1) / (count_right[token_1] + vocab_number)
                prob_dict.setdefault(token, value)
            except KeyError:
                continue
        return prob_dict

    def type_recogniction(self,candidate, wrong_word):
        wrong_types = Levenshtein.editops(candidate, wrong_word)
        right_list = list(candidate)
        wrong_list = list(wrong_word)
        token_list = []
        ban_list = []
        for i in range(len(wrong_types)):
            if i in ban_list:
                continue
            wrong_type = wrong_types[i]
            if i + 1 < len(wrong_types):
                wrong_type_2 = wrong_types[i + 1]
                right_id_1 = int(wrong_type[1])
                right_id_2 = int(wrong_type_2[1])
                wrong_id_1 = int(wrong_type[2])
                wrong_id_2 = int(wrong_type_2[2])
                if wrong_type[0] == 'replace' and wrong_types[i + 1][0] == 'replace':
                    right_token = right_list[right_id_1] + right_list[right_id_2]
                    wrong_token = wrong_list[wrong_id_2] + wrong_list[wrong_id_1]
                else:
                    right_token = 1
                    wrong_token = 0
                if wrong_type[0] == 'replace' and wrong_types[i + 1][
                    0] == 'replace' and right_id_1 + 1 == right_id_2 and right_token == wrong_token:
                    token = right_token + '|' + wrong_list[wrong_id_1] + wrong_list[wrong_id_2]
                    ban_list.append(i+1)
                else:
                    right_id_1 = int(wrong_type[1])
                    wrong_id_1 = int(wrong_type[2])
                    if wrong_type[0] == 'insert':
                        if right_id_1 != 0:
                            token = wrong_list[wrong_id_1 - 1] + '|' + wrong_list[wrong_id_1 - 1] + wrong_list[
                                wrong_id_1]
                        elif right_id_1 == 0:
                            token = '>' + '|' + '>' + wrong_list[wrong_id_1]
                    elif wrong_type[0] == 'delete':
                        if wrong_id_1 != 0:
                            token = right_list[right_id_1 - 1] + right_list[right_id_1] + '|' + right_list[
                                right_id_1 - 1]
                        elif wrong_id_1 == 0:
                            token = '>' + right_list[right_id_1] + '|' + '>'
                    elif wrong_type[0] == 'replace':
                        token = right_list[right_id_1] + '|' + wrong_list[wrong_id_1]
            else:
                right_id_1 = int(wrong_type[1])
                wrong_id_1 = int(wrong_type[2])
                if wrong_type[0] == 'insert':
                    if right_id_1 != 0:
                        token = wrong_list[wrong_id_1 - 1] + '|' + wrong_list[wrong_id_1 - 1] + wrong_list[
                            wrong_id_1]
                    elif right_id_1 == 0:
                        token = '>' + '|' + '>' + wrong_list[wrong_id_1]
                elif wrong_type[0] == 'delete':
                    if wrong_id_1 != 0:
                        token = right_list[right_id_1 - 1] + right_list[right_id_1] + '|' + right_list[
                            right_id_1 - 1]
                    elif wrong_id_1 == 0:
                        token = '>' + right_list[right_id_1] + '|' + '>'
                elif wrong_type[0] == 'replace':
                    token = right_list[right_id_1] + '|' + wrong_list[wrong_id_1]
            token_list.append(token)
        return token_list

    def noisy_channel(self, bigram_dict, word_list, error_count, error_number):
        candidate_dict = {}
        prob_dict = {}
        word_list_corrected = word_list
        if error_count < error_number:
            for i in range(len(word_list)):
                if word_list[i] in self.word_set:
                    candidate_list = []
                    for candidate in self.word_set:
                        distance = Levenshtein.distance(word_list[i], candidate)
                        if distance == 1:
                            candidate_list.append(candidate)
                    candidate_dict.setdefault(i, candidate_list)
            for id, candidate_list in candidate_dict.items():
                word_list_new =['</s>'] + word_list
                for j in range(len(candidate_list)):
                    word_list_new[id+1] = candidate_list[j]
                    word_list_2 = [str(word_list_new[i])+' '+str(word_list_new[i+1]) for i in range(len(word_list_new)-1)]
                    probability = 0
                    for token in word_list_2:
                        try:
                            probability += math.log10(bigram_dict[token])
                        except KeyError:
                            probability += -15
                    prob_dict.setdefault((id,j),probability)
            prob_sorted = sorted(prob_dict.items(),key=lambda x:x[1], reverse=True)
            id = prob_sorted[0][0][0]
            j = prob_sorted[0][0][1]
            word_list_corrected[id] = candidate_dict[id][j]
        return word_list_corrected






    def auto_correction(self, num2error, num2sent, prob_dict, ngram_prob_dict,n):
        num2sent_corrected = {}
        error_index_count = 0
        error_candidate_count = 0
        for id, sentence in num2sent.items():
            word_list = nltk.word_tokenize(sentence)
            word_list_with_head = ['<s>'] + word_list
            error_count = 0
            candidate_dict = {}
            for i in range(len(word_list)):
                word = word_list[i]
                if word not in self.word_set:
                    error_count += 1
                    candidate_list = []
                    for candidate in self.word_set:
                        if editdistance.eval(candidate, word) < 3:
                            candidate_list.append(candidate)
                    candidate_dict.setdefault(i,candidate_list)
            if error_count != int(num2error[id]):
                error_candidate_count += 1
                word_list = self.noisy_channel(bigram_dict=ngram_prob_dict, word_list=word_list, error_count=error_count, error_number=int(num2error[id]))
                print(id)
                #print('Error count: ',error_count)
                #print('Error: ', num2error[id])
                #print('Sentence id: ', id)

            for word_id, candidate_list in candidate_dict.items():
                candidate_prob_dict = {}
                for candidate in candidate_list:
                    token_list = self.type_recogniction(candidate=candidate,wrong_word=word_list[word_id])
                    if n == 2:
                        word_tuple = word_list_with_head[word_id] + ' ' + candidate
                        try:
                            prob = math.log10(ngram_prob_dict[word_tuple])
                        except KeyError:
                            error_index_count += 1
                            prob = -15
                    elif n == 1:
                        word_tuple = candidate
                        try:
                            prob = math.log10(ngram_prob_dict[word_tuple])
                        except KeyError:
                            error_index_count += 1
                            # print(word_tuple)
                            prob = -15
                    elif n == 3:
                        word_tuple = word_list_with_head[word_id-1]  + ' ' + word_list_with_head[word_id] + ' ' + candidate
                        try:
                            prob = math.log10(ngram_prob_dict[word_tuple])
                        except KeyError:
                            #print(word_tuple)
                            error_index_count += 1
                            prob = -15

                    for token in token_list:
                        try:
                            prob = prob + math.log10(prob_dict[token])
                        except KeyError:
                            #print(token)
                            prob = prob - 15
                    candidate_prob_dict.setdefault(candidate, prob)
                candidate_prob = sorted(candidate_prob_dict.items(),key=lambda x:x[1], reverse=True)
                try:
                    word_corrected = candidate_prob[0][0]
                except IndexError:
                    print(candidate_prob)
                #print(id, word_list[word_id], word_corrected, candidate_prob)
                word_list[word_id] = word_corrected
            sentence_corrected = " ".join(word_list)
            num2sent_corrected.setdefault(id, sentence_corrected)
        print('Error Index: ', error_index_count)
        print('Number of error candidate: ', error_candidate_count)
        return num2sent_corrected

    def auto_correction_backoff(self, num2error, num2sent, prob_dict, unigram_prob_dict, bigram_prob_dict,trigram_prob_dict):
        num2sent_corrected = {}
        error_index_count = 0
        error_candidate_count = 0
        for id, sentence in num2sent.items():
            word_list = nltk.word_tokenize(sentence)
            word_list_with_head = ['>'] + word_list
            error_count = 0
            candidate_dict = {}
            for i in range(len(word_list)):
                word = word_list[i]
                if word not in self.word_set:
                    error_count += 1
                    candidate_list = []
                    for candidate in self.word_set:
                        if editdistance.eval(candidate, word) < 3:
                            candidate_list.append(candidate)
                    candidate_dict.setdefault(i, candidate_list)
            if error_count != int(num2error[id]):
                error_candidate_count += 1

                # print('Error count: ',error_count)
                # print('Error: ', num2error[id])
                # print('Sentence id: ', id)

            for word_id, candidate_list in candidate_dict.items():
                candidate_prob_dict = {}
                for candidate in candidate_list:
                    token_list = self.type_recogniction(candidate=candidate, wrong_word=word_list[word_id])
                    word_tuple = word_list_with_head[word_id-1] + word_list_with_head[word_id] + ' ' + candidate
                    try:
                        prob = math.log10(trigram_prob_dict[word_tuple])
                    except KeyError:
                        try:
                            word_tuple = word_list_with_head[word_id] + ' ' + candidate
                            prob = math.log10(bigram_prob_dict[word_tuple])
                        except KeyError:
                            word_tuple = candidate
                            prob = math.log10(unigram_prob_dict[word_tuple])
                        error_index_count += 1
                        prob = -15


                    for token in token_list:
                        try:
                            prob = prob + math.log10(prob_dict[token])
                        except KeyError:
                            prob = prob - 15
                    candidate_prob_dict.setdefault(candidate, prob)
                candidate_prob = sorted(candidate_prob_dict.items(), key=lambda x: x[1], reverse=True)
                try:
                    word_corrected = candidate_prob[0][0]
                except IndexError:
                    print(candidate_prob)
                word_list[word_id] = word_corrected
            sentence_corrected = " ".join(word_list)
            num2sent_corrected.setdefault(id, sentence_corrected)
        print('Error Index: ', error_index_count)
        print('Number of error candidate: ', error_candidate_count)
        return num2sent_corrected








model1 = spelling_correction()
#count_dict, prob_dict = model1.confusion_matrix()
prob_dict = model1.confusion_matrix_new_smoothing()


book_word = reuters.words()
#book_word = [word.lower() for word in book_word]
book_sent = reuters.sents()
#book_sent = [[word.lower() for word in sent] for sent in book_sent]
model = lm.toy_LM()
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
unigram_prob_dict = model.good_turing_unigram(unigram_dict=unigram_dict,
                                              word_number=word_number)
bigram_prob_dict = model.add1_smoothed_bigram(unigram_dict=unigram_dict,
                                           ngram_dict=bigram_dict,
                                           sentence_number=sentence_number)
trigram_prob_dict = model.add1_smoothed_trigram(bigram_dict=bigram_dict,
                                           trigram_dict=trigram_dict)

model1.word_set_get()
num2error, num2sent = model1.data_read()
print('Start correct')



num2sent_corrected_unigram = model1.auto_correction(num2sent=num2sent,
                                            num2error=num2error,
                                            ngram_prob_dict=bigram_prob_dict,
                                            n = 2,
                                            prob_dict=prob_dict
)

with open('result.txt', mode='w',encoding='utf-8') as f:
    for num,sent in num2sent_corrected_unigram.items():
        thisline = num + '\t' + sent + '\n'
        f.write(thisline)



'''
unigram_prob_dict = model.good_turing_unigram(unigram_dict=unigram_dict,
                                              word_number=word_number)
bigram_prob_dict = model.add1_smoothed_bigram(unigram_dict=unigram_dict,
                                           ngram_dict=bigram_dict,
                                           sentence_number=sentence_number)
trigram_prob_dict = model.add1_smoothed_trigram(bigram_dict=bigram_dict,
                                           trigram_dict=trigram_dict)
'''
