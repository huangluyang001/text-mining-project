import codecs
import nltk
import jieba
import json
import math
import pickle


f = codecs.open('news.txt','r','utf-8')
news = [eval(i) for i in f.readlines()]
f.close()
f = open('train.txt','r')
trains = [i.split() for i in f.readlines()]
f.close()
f = open('test.txt','r')
tests = [i.split() for i in f.readlines()]
f.close()

class Preprocess():
    def __init__(self):
        self.id2title = {}
        self.id2content = {}
        self.id2label = {}


    @staticmethod
    def segment_one_sentence(sent):
        return jieba.lcut(sent)

    def segmentation(self):
        with open('news_segmented.json',mode='w',encoding='utf-8') as f:
            for news_dict in news:
                sent_1 = news_dict['content']
                sent_2 = news_dict['title']
                sents = sent_1.split('\n')
                word_content = []
                for sent in sents:
                    word_content += self.segment_one_sentence(sent)
                news_dict.setdefault('Token_content',word_content)
                word_title = self.segment_one_sentence(sent_2)
                news_dict.setdefault('Token_title',word_title)
                f.write(json.dumps(news_dict, ensure_ascii=False))
                f.write('\n')

    def read_x_y(self):
        with open('news_segmented.json',encoding='utf-8') as f:
            for line in f:
                news_dict = json.loads(line)
                id = int(news_dict['id'])
                title_token = news_dict['Token_title']
                content_token = news_dict['Token_content']
                self.id2content.setdefault(id, content_token)
                self.id2title.setdefault(id, title_token)
        for train in trains:
            ids = train[1].split(',')
            for id in ids:
                self.id2label.setdefault(int(id),train[0])


class Naive_Bayes_Classifier(Preprocess):
    def __init__(self):
        Preprocess.__init__(self)
        Preprocess.read_x_y(self)
        self.p_c = {}
        self.p_wc = {}
        self.t_p_c = {}
        self.t_p_wc = {}
        self.c_p_c = {}
        self.c_p_wc = {}
        self.word_set = set()

    def train(self, normalization = 'Add-1', debug = 'off'):
        word_set = set()
        title_count_dict = {}
        content_count_dict = {}
        positive = 0
        negative = 0
        for id ,words in self.id2title.items():
            try:
                label = self.id2label[id]
            except KeyError:
                continue
            for word in words:
                word_set.add(word)
                token = str(word) + '|' + label
                try:
                    title_count_dict[token] += 1
                except:
                    title_count_dict.setdefault(token, 1)
                if label == '+1':
                    positive += 1
                elif label == '-1':
                    negative += 1
                else:
                    print(label)
        for id ,words in self.id2content.items():
            try:
                label = self.id2label[id]
            except KeyError:
                continue
            for word in words:
                word_set.add(word)
                token = str(word) + '|' + label
                try:
                    content_count_dict[token] += 1
                except:
                    content_count_dict.setdefault(token, 1)
                if label == '+1':
                    positive += 1
                elif label == '-1':
                    negative += 1
                else:
                    print(label)
        for word in word_set:
            token = str(word) + '|' + '+1'
            title_count_dict.setdefault(token, 0)
            content_count_dict.setdefault(token, 0)
            token = str(word) + '|' + '-1'
            title_count_dict.setdefault(token, 0)
            content_count_dict.setdefault(token, 0)

        # normalization
        if normalization == 'Add-1':
            word_number = len(word_set)
            if debug == 'on':
                print(word_number)
            for token, count in title_count_dict.items():
                label = token.split('|')[1]
                if label == '+1':
                    prob = (count + 1) / (positive + word_number)
                    self.t_p_wc.setdefault(token, prob)
                elif label == '-1':
                    prob = (count + 1) / (negative + word_number)
                    self.t_p_wc.setdefault(token, prob)
                else:
                    print(token)
            for token, count in content_count_dict.items():
                label = token.split('|')[1]
                if label == '+1':
                    prob = (count + 1) / (positive + word_number)
                    self.c_p_wc.setdefault(token, prob)
                elif label == '-1':
                    prob = (count + 1) / (negative + word_number)
                    self.c_p_wc.setdefault(token, prob)
                else:
                    print(token)
            p_positive = positive / (positive + negative)
            p_negative = negative / (positive + negative)
            self.p_c.setdefault(1, p_positive)
            self.p_c.setdefault(-1, p_negative)
            self.word_set = word_set

    def prediction(self,type='MajorityVoting'):
        for test in tests:
            ids = test[1].split(',')
            id2prob_positive = {}
            id2prob_negative = {}
            positive_bool = 0
            negative_bool = 0
            p_t = 0
            p_c = 0
            f_c = 0
            f_t = 0
            for id in ids:
                word_title = self.id2title[int(id)]
                word_content = self.id2content[int(id)]
                word_all = word_title + word_content
                prob_positive_t = math.log10(self.p_c[1])
                prob_negative_t = math.log10(self.p_c[-1])
                prob_positive_c = math.log10(self.p_c[1])
                prob_negative_c = math.log10(self.p_c[-1])
                for word in word_all:
                    if word in self.word_set:
                        token_positive = str(word) + '|' + '+1'
                        token_negative = str(word) + '|' + '-1'
                        try:
                            prob_positive_t += math.log10(self.t_p_wc[token_positive])
                            prob_negative_t += math.log10(self.t_p_wc[token_negative])
                            prob_positive_c += math.log10(self.c_p_wc[token_positive])
                            prob_negative_c += math.log10(self.c_p_wc[token_negative])
                        except KeyError:
                            print(token_positive,token_negative)
                    else:
                        continue
                p_t += prob_positive_t
                p_c += prob_positive_c
                f_t += prob_negative_t
                f_c += prob_negative_c
                if prob_positive_c > prob_negative_c:
                    positive_bool += 1
                else:
                    negative_bool += 1
                if prob_positive_t > prob_negative_t:
                    positive_bool += 1
                else:
                    negative_bool += 1
                id2prob_positive.setdefault(int(id), prob_positive_t)
                id2prob_negative.setdefault(int(id), prob_negative_t)
            if type == 'Fusion':
                if p_c > f_c and p_t > f_t:
                    test[0] = '+1'
                elif p_c < f_c and p_t < f_t:
                    test[0] = '-1'
                elif p_c + p_t > f_c + f_t:
                    test[0] = '+1'
                else:
                    test[0] = '-1'
            elif type == 'MajorityVoting':
                if positive_bool > negative_bool:
                    test[0] = '+1'
                elif positive_bool < negative_bool:
                    test[0] = '-1'
                else:
                    test[0] = '+1'
                    print('Same',positive_bool,negative_bool)
        with open('result.txt','w',encoding='utf-8') as f:
            for test in tests:
                thisline = test[0] + '\t' + test[1] + '\n'
                f.write(thisline)


class Other_classifiers(Preprocess):
    def __init__(self):
        Preprocess.__init__(self)
        Preprocess.read_x_y(self)
        self.p_c = {}
        self.p_wc = {}
        self.word_set = set()
        self.ly_adj_set = set()

    def adj_set_load(self):
        with open('负面评价词语（中文）.txt',encoding='gbk') as f:
            for i in range(2):
                f.readline()
            for line in f:
                self.ly_adj_set.add(line.strip())
        with open('负面情感词语（中文）.txt',encoding='gbk') as f:
            for i in range(2):
                f.readline()
            for line in f:
                self.ly_adj_set.add(line.strip())
        with open('正面评价词语（中文）.txt',encoding='gbk') as f:
            for i in range(2):
                f.readline()
            for line in f:
                self.ly_adj_set.add(line.strip())
        with open('正面情感词语（中文）.txt',encoding='gbk') as f:
            for i in range(2):
                f.readline()
            for line in f:
                self.ly_adj_set.add(line.strip())

    @staticmethod
    def word_feature(content, word_features):
        words = set(content)
        features = {}
        for (word, freq) in word_features:
            features.setdefault(word, word in words)
        return features





    def feature_extract(self):
        self.adj_set_load()
        all_words = nltk.FreqDist(word for id, word_list in self.id2content.items() for word in word_list)
        word_features = all_words.most_common(100)
        all_adjs = nltk.FreqDist(word for id, word_list in self.id2content.items() for word in word_list if word in self.ly_adj_set)
        adj_features = all_adjs.most_common(100)
        train_set_normal = [(self.word_feature(self.id2content[ly_id],word_features),label) for ly_id, label in self.id2label.items()]
        train_set_adj = [(self.word_feature(self.id2content[ly_id],adj_features),label) for ly_id, label in self.id2label.items()]
        train_set_all = [(self.word_feature(self.id2content[ly_id],word_features+adj_features),label) for ly_id, label in self.id2label.items()]


        return train_set_all, word_features


    def train(self,classifier_type='DecisionTree'):
        train_set, word_features = self.feature_extract()
        if classifier_type == 'NaiveBayes':
            classifier = nltk.classify.NaiveBayesClassifier.train(train_set)
        elif classifier_type == 'DecisionTree':
            classifier = nltk.classify.DecisionTreeClassifier.train(train_set)
        elif classifier_type == 'maxentropy':
            classifier = nltk.classify.MaxentClassifier.train(train_set,trace=3,max_iter=5)
        else:
            print('Wrong Classifier Type')

        testone = nltk.classify.accuracy(classifier,train_set)
        print(testone)
        return classifier, word_features

    def prediction(self,classifier, word_features):
        for test in tests:
            p = 0
            n = 0
            for ly_id in test[1].split(','):
                feature4test = self.word_feature(self.id2content[int(ly_id)],word_features)
                label = classifier.classify(feature4test)
                if label == '+1':
                    p += 1
                elif label == '-1':
                    n += 1
                else:
                    print(label)
            if p > n:
                test[0] = '+1'
            elif p < n:
                test[0] = '-1'
            else:
                print(test)
                test[0] = '+1'
        with open('result.txt','w',encoding='utf-8') as f:
            for test in tests:
                thisline = test[0] + '\t' + test[1] + '\n'
                f.write(thisline)










'''
#分词，仅需使用一次
preprocess = Preprocess()
preprocess.segmentation()
'''

#自己实现的朴素贝叶斯
model = Naive_Bayes_Classifier()
model.train()
model.prediction(type='Fusion')

'''
model = Other_classifiers()
classifier, word_feature = model.train(classifier_type='maxentropy') #classifier_type: 'NaiveBayes','DecisionTree','maxentropy'
save_classifier = open("maxent.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
model.prediction(classifier, word_feature)
'''

