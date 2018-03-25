import math
import os
from pyltp import Postagger


class HMM():
    def __init__(self):
        pass

    @staticmethod
    def Probability_calculation(dict1, dict2):
        prob_dict = {}
        for token, value in dict1.items():
            token_list = token.split('|')
            token_list.pop()
            word = '|'.join(token_list)
            prob = value / dict2[word]
            prob_dict.setdefault(token, prob)
        return prob_dict

    def Generate_prob(self, filename):
        with open(filename, encoding='utf-8') as f:
            words = []
            tags = []
            word_dict = {}
            tag_dict = {}
            word_tag_dict = {}
            zero_tag = ['*','*']
            trigram_dict = {}
            bigram_dict = {}
            for line in f:
                if len(line.strip()) == 0:
                    for i in range(len(words)):
                        try:
                            word_dict[words[i]] += 1
                        except KeyError:
                            word_dict.setdefault(words[i], 1)
                        try:
                            tag_dict[tags[i]] += 1
                        except KeyError:
                            tag_dict.setdefault(tags[i], 1)
                        try:
                            token = tags[i] + '|' + words[i]
                            word_tag_dict[token] += 1
                        except KeyError:
                            token = tags[i] + '|' + words[i]
                            word_tag_dict.setdefault(token, 1)
                    tags = zero_tag + tags + ['</s>']
                    tag_bigram = [tags[i] + '|' + tags[i+1] for i in range(len(tags)-1)]
                    tag_trigram = [tags[i] + '|' + tags[i+1] + '|' + tags[i+2] for i in range(len(tags)-2)]
                    for token in tag_trigram:
                        try:
                            trigram_dict[token] += 1
                        except KeyError:
                            trigram_dict.setdefault(token, 1)
                    for token in tag_bigram:
                        try:
                            bigram_dict[token] += 1
                        except KeyError:
                            bigram_dict.setdefault(token, 1)
                    words = []
                    tags = []
                else:
                    data = line.strip().split('\t')
                    words.append(data[0])
                    tags.append(data[1])
        word_token_prob = self.Probability_calculation(word_tag_dict, tag_dict)
        token_trigram_prob = self.Probability_calculation(trigram_dict, bigram_dict)
        return word_token_prob, token_trigram_prob, tag_dict

    def Generate_prob_postag(self, filename):
        with open(filename, encoding='utf-8') as f:
            words = []
            tags = []
            postags = []
            word_dict = {}
            postag_tag_dict = {}
            tag_dict = {}
            word_tag_dict = {}
            zero_tag = ['*','*']
            trigram_dict = {}
            bigram_dict = {}
            for line in f:
                if len(line.strip()) == 0:
                    for i in range(len(words)):
                        try:
                            word_dict[words[i]] += 1
                        except KeyError:
                            word_dict.setdefault(words[i], 1)
                        try:
                            tag_dict[tags[i]] += 1
                        except KeyError:
                            tag_dict.setdefault(tags[i], 1)
                        try:
                            token = tags[i] + '|' + words[i]
                            word_tag_dict[token] += 1
                        except KeyError:
                            token = tags[i] + '|' + words[i]
                            word_tag_dict.setdefault(token, 1)
                        try:
                            token = tags[i] + '|' + postags[i]
                            postag_tag_dict[token] += 1
                        except KeyError:
                            token = tags[i] + '|' + postags[i]
                            postag_tag_dict.setdefault(token, 1)
                    tags = zero_tag + tags + ['</s>']
                    tag_bigram = [tags[i] + '|' + tags[i+1] for i in range(len(tags)-1)]
                    tag_trigram = [tags[i] + '|' + tags[i+1] + '|' + tags[i+2] for i in range(len(tags)-2)]
                    for token in tag_trigram:
                        try:
                            trigram_dict[token] += 1
                        except KeyError:
                            trigram_dict.setdefault(token, 1)
                    for token in tag_bigram:
                        try:
                            bigram_dict[token] += 1
                        except KeyError:
                            bigram_dict.setdefault(token, 1)
                    words = []
                    tags = []
                    postags = []
                else:
                    data = line.strip().split('\t')
                    words.append(data[0])
                    postags.append(data[1])
                    tags.append(data[2])
        word_token_prob = self.Probability_calculation(word_tag_dict, tag_dict)
        token_trigram_prob = self.Probability_calculation(trigram_dict, bigram_dict)
        postag_token_prob = self.Probability_calculation(postag_tag_dict, tag_dict)
        return word_token_prob, token_trigram_prob, postag_token_prob, tag_dict


    def predict(self, word_token_prob, token_trigram_prob, token_dict, filename, output_filename):
        with open(filename, encoding='utf-8') as f:
            with open(output_filename, encoding='utf-8', mode = 'w') as g:
                words = []
                tokens = ['*','*']
                prob_dict = {}
                true_tags = []
                for line in f:
                    if len(line.strip()) == 0:
                        for word in words:
                            for token, value in token_dict.items():
                                word_token = token + '|' + word
                                token_trigram = tokens[-2] + '|' + tokens[-1] + '|' + token
                                try:
                                    aaa = word_token_prob[word_token]
                                    ly_prob = math.log10(aaa)
                                except KeyError:
                                    ly_prob =  -15
                                try:
                                    ly_prob += math.log10(token_trigram_prob[token_trigram])
                                except KeyError:
                                    ly_prob += -15
                                prob_dict.setdefault(token, ly_prob)
                            prob_sorted = sorted(prob_dict.items(), key=lambda x:x[1], reverse=True)
                            prob_dict = {}
                            tokens.append(prob_sorted[0][0])
                        for i in range(len(words)):
                            thisline = words[i] + '\t' + tokens[i+2] + '\t' + true_tags[i]+ '\n'
                            g.write(thisline)
                        g.write('\n')
                        words = []
                        tokens = ['*','*']
                        true_tags = []
                    else:
                        words.append(line.strip().split()[0])
                        true_tags.append(line.strip().split()[1])

    def predict_postag(self, word_token_prob, token_trigram_prob, postag_token_prob, token_dict, filename, output_filename):
        with open(filename, encoding='utf-8') as f:
            with open(output_filename, encoding='utf-8', mode = 'w') as g:
                words = []
                postags = []
                tokens = ['*','*']
                prob_dict = {}
                true_tags = []
                for line in f:
                    if len(line.strip()) == 0:
                        for i in range(len(words)):
                            for token, value in token_dict.items():
                                word_token = token + '|' + words[i]
                                token_trigram = tokens[-2] + '|' + tokens[-1] + '|' + token
                                postag_token = token + '|' + postags[i]
                                try:
                                    aaa = word_token_prob[word_token]
                                    ly_prob = math.log10(aaa)
                                except KeyError:
                                    ly_prob =  -15
                                try:
                                    ly_prob += math.log10(token_trigram_prob[token_trigram])
                                except KeyError:
                                    ly_prob += -15
                                try:
                                    ly_prob += math.log10(postag_token_prob[postag_token])
                                except KeyError:
                                    ly_prob +=  -8
                                prob_dict.setdefault(token, ly_prob)
                            prob_sorted = sorted(prob_dict.items(), key=lambda x:x[1], reverse=True)
                            prob_dict = {}
                            tokens.append(prob_sorted[0][0])
                        for i in range(len(words)):
                            thisline = words[i] + '\t' + tokens[i+2] + '\t' + true_tags[i]+ '\n'
                            g.write(thisline)
                        g.write('\n')
                        words = []
                        tokens = ['*','*']
                        true_tags = []
                        postags = []
                    else:
                        words.append(line.strip().split()[0])
                        postags.append(line.strip().split()[1])
                        true_tags.append(line.strip().split()[2])





def pos_tagging(filename, output_filename):
    LTP_DATA_DIR = 'ltp_data_v3.4.0'  # ltp模型目录的路径
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
    postagger = Postagger()  # 初始化实例
    postagger.load(pos_model_path)  # 加载模型
    with open(filename, encoding='utf-8') as f:
        with open(output_filename, encoding='utf-8', mode='w') as g:
            words = []
            tags = []
            for line in f:
                if len(line.strip()) == 0:
                    postags = postagger.postag(words)  # 词性标注
                    pos_tags = list(postags)
                    for i in range(len(words)):
                        thisline = words[i] + '\t' + pos_tags[i] + '\t' + tags[i] + '\n'
                        g.write(thisline)
                    g.write('\n')
                    words = []
                    tags = []
                else:
                    words.append(line.strip().split()[0])
                    tags.append(line.strip().split()[1])






if __name__ == '__main__':
    pos_tagging(filename='argument_train.txt', output_filename='argument_train_processed.txt')
    pos_tagging(filename='argument_test.txt', output_filename='argument_test_processed.txt')
    model = HMM()
    word_token_prob, token_trigram_prob, postag_token_prob, token_dict = model.Generate_prob_postag(
        filename='argument_train_processed.txt')
    model.predict_postag(word_token_prob=word_token_prob, token_trigram_prob=token_trigram_prob, token_dict=token_dict,
                         filename='argument_test_processed.txt', output_filename='argument_result.txt',
                         postag_token_prob=postag_token_prob)

'''
    pos_tagging(filename='trigger_test.txt', output_filename='trigger_test_processed.txt')
    model = HMM()
    word_token_prob, token_trigram_prob, token_dict = model.Generate_prob(filename='trigger_train.txt')
    model.predict(word_token_prob=word_token_prob, token_trigram_prob=token_trigram_prob, token_dict=token_dict,
                  filename='trigger_test.txt', output_filename='trigger_result.txt')
    word_token_prob, token_trigram_prob, token_dict = model.Generate_prob(filename='argument_train.txt')
    model.predict(word_token_prob=word_token_prob, token_trigram_prob=token_trigram_prob, token_dict=token_dict,
                  filename='argument_test.txt', output_filename='argument_result.txt')
'''

