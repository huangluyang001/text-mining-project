import glob
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import nltk
import re

class Preprocessing():
    def __init__(self):
        pass

    def load_dataset(self):
        episodeid = 0
        paraid = 0
        paraid2scene = {}
        paraid2sents = {}
        paraid2chars = {}
        episodeid2paraid = {}
        para_list = []
        characters = set()
        word_dict = set()
        for f in glob.glob('./4_character_dataset/*_*.txt'):
            with open(f, encoding='utf-8') as g:
                for line in g:
                    data = line.strip().split(':')
                    if len(data) > 1:
                        char = data[0].split('(')[0]
                        char = char.lower()
                        char = char.strip()
                        if char == 'like this':
                            continue
                        sent = data[1:]
                        pieces = ''
                        for piece in sent:
                            pieces += piece
                        sent = pieces.strip()
                        sent = nltk.word_tokenize(sent)
                        for i in range(len(sent)):
                            word = sent[i]
                            word = word.lower()
                            sent[i] = word
                            word_dict.add(word)
                        if char == 'scene' and paraid != 0:
                            paraid2sents.setdefault(paraid, sents)
                            paraid2chars.setdefault(paraid, chars)
                            sents = []
                            chars = []
                            para_list.append(paraid)
                            paraid += 1
                            paraid2scene.setdefault(paraid, sent)
                        elif char == 'scene' and paraid == 0:
                            paraid += 1
                            paraid2scene.setdefault(paraid, sent)
                            sents = []
                            chars = []
                        else:
                            sents.append(sent)
                            chars.append(char)
                            characters.add(char)
                    else:
                        continue
                episodeid2paraid.setdefault(episodeid, para_list)
                para_list = []
                episodeid += 1
        paraid2sents.setdefault(paraid, sents)
        paraid2chars.setdefault(paraid, chars)
        para_number = paraid
        return characters, paraid2scene, paraid2chars, paraid2sents, episodeid2paraid, para_number, word_dict

    def dataset_count(self,characters, paraid2scene, paraid2chars, paraid2sents, episodeid2paraid, para_number):
        scene_count_dict = {}
        line_count_dict = {}
        episode_count_dict = {}
        total_scene = 0
        total_episode = 0
        total_line = 0
        for character in characters:
            scene_count_dict.setdefault(character,0)
            line_count_dict.setdefault(character,0)
            episode_count_dict.setdefault(character,0)
        for episode_id, para_ids in episodeid2paraid.items():
            total_episode += 1
            for character in characters:
                for paraid in para_ids:
                    if character in paraid2chars[paraid]:
                        episode_count_dict[character] += 1
                        break

        for paraid,chars in paraid2chars.items():
            total_scene += 1
            for character in characters:
                if character in chars:
                    scene_count_dict[character] += 1


            for char in chars:
                line_count_dict[char] += 1
                total_line += 1

        print('Total episodes: ',total_episode)
        print('Total scenes: ', total_scene)
        print('Total lines: ', total_line)
        '''
        with open('character_count.txt',mode='w',encoding='utf-8') as f:
            thisline = 'char\tepisode\tscene\tline\n'
            f.write(thisline)
            for character in characters:
                thisline = '%s\t%s\t%s\t%s\n' %(character,episode_count_dict[character],scene_count_dict[character],line_count_dict[character])
                f.write(thisline)
        '''





    def encoding(self, characters, word_dict):
        char2id = {}
        id2char = {}
        word2id = {}
        id2word = {}
        id = 0
        for character in characters:
            char2id.setdefault(character, id)
            id2char.setdefault(id, character)
            id += 1
        char_number = id
        id = 0
        word2id.setdefault('<unk>',id)
        id2word.setdefault(id, '<unk>')
        id += 1
        for word in word_dict:
            word2id.setdefault(word, id)
            id2word.setdefault(id, word)
            id += 1
        word_number = id
        return char2id, id2char, word2id, id2word, char_number, word_number

    def encoding_reduction(self, characters, word_dict): #reduce character space
        char2id = {}
        id2char = {}
        main_character_list = ['sheldon','leonard','penny','howard','raj','bernadette','amy','stuart']
        char2id.setdefault('other', 0)
        id2char.setdefault(0, 'other')
        id = 1
        for character in main_character_list:
            assert character in characters
            char2id.setdefault(character, id)
            id2char.setdefault(id, character)
            id += 1
        char_number = id
        word2id, id2word, embedding_matrix, word_number = self.load_embedding(word_dict)
        return char2id, id2char, word2id, id2word, char_number, word_number, embedding_matrix

    def encoding_reduction_onlychar(self, characters, word_dict): #reduce character space abandoned
        char2id = {}
        id2char = {}
        word2id = {}
        id2word = {}
        main_character_list = ['sheldon','leonard','penny','howard','raj','bernadette','amy','stuart']
        char2id.setdefault('other', 0)
        id2char.setdefault(0, 'other')
        id = 1
        for character in main_character_list:
            assert character in characters
            char2id.setdefault(character, id)
            id2char.setdefault(id, character)
            id += 1
        char_number = id
        id = 0
        word2id.setdefault('<unk>',id)
        id2word.setdefault(id, '<unk>')
        id += 1
        for word in word_dict:
            word2id.setdefault(word, id)
            id2word.setdefault(id, word)
            id += 1
        word_number = id
        return char2id, id2char, word2id, id2word, char_number, word_number

    def load_embedding(self, word_dict, add_char = True, emotion = True):
        word_dict.add('<unk>')
        word2embedding = {}
        word2id = {}
        id2word = {}
        id2embedding = {}
        word2id.setdefault('<unk>',0)
        id2word.setdefault(0,'<unk>')
        word_set = set()
        id = 1
        word2vector = self.feature_read()
        zero_vector = np.zeros(12)
        with open('glove.6B.50d.txt',encoding='utf-8') as f:
            for line in f:
                data = line.strip().split(' ')
                word = data[0]
                embedding = data[1:]
                word_set.add(word)
                if word in word_dict:
                    if len(embedding) == 50:
                        vector = word2vector.get(word, zero_vector)
                        embedding = np.array(embedding)
                        embedding_new = np.concatenate((embedding,vector))
                        if emotion == True:
                            word2embedding.setdefault(word, embedding_new)
                        else:
                            word2embedding.setdefault(word, embedding)
                        word2id.setdefault(word, id)
                        id2word.setdefault(id, word)
                        id += 1
                    else:
                        continue
        word_number = len(word2id)
        embedding_dim = len(word2embedding['sheldon'])
        for word,embedding in word2embedding.items():
            id = word2id[word]
            id2embedding.setdefault(id, embedding)
        embedding_number = len(id2embedding)
        if add_char == False:
            assert word_number == embedding_number
            embedding_matrix = np.zeros([word_number,embedding_dim])
            for id,embedding in id2embedding.items():
                embedding_matrix[id,:] = embedding
            print('Word number: ', word_number)
        else:
            main_character_list = ['sheldon', 'leonard', 'penny', 'howard', 'raj', 'bernadette', 'amy', 'stuart']
            embedding_matrix = np.zeros([word_number,embedding_dim])
            for id,embedding in id2embedding.items():
                embedding_matrix[id,:] = embedding
            for i in range(len(main_character_list)):
                j = 0
                if main_character_list[i] not in word2id:
                    index = embedding_number + j
                    j += 1
                    embedding_matrix[index,:] = np.random.randn(embedding_dim)
                    word2id.setdefault(main_character_list[i], index)
                    id2word.setdefault(index, main_character_list[i])
            word_number += j
            print('Word number: ', word_number)
            print('Embedding Dim:', embedding_dim)


        '''
        for word in word_dict:
            if word in word_set:
                continue
            else:
                print(word)
        '''

        return word2id,id2word,embedding_matrix,word_number


    def feature_read(self):
        word_set = set()
        word2vector = {}
        with open('word_feature_cleaned.txt', mode='r', encoding='utf-8') as f:
            for i in range(1):
                f.readline()
            for line in f:
                data = line.strip().split('\t')
                word = data[0].lower()
                word_set.add(word)
                features = data[1:]
                for i in range(len(features)):
                    if features[i] == '-':
                        features[i] = 0.0
                    else:
                        features[i] = float(features[i])
                feature_vector = np.array(features,dtype='float')

                feature_vector = feature_vector

                assert feature_vector.shape[0] == 12
                word2vector.setdefault(word, feature_vector)
        return word2vector









    def generate_X_Y(self, characters, paraid2scene, paraid2chars, paraid2sents, episodeid2paraid, para_number, word_dict):
        char2id, id2char, word2id, id2word, char_number, word_number, _ = self.encoding_reduction(characters, word_dict)
        X_scene = []
        X_current_sents = []
        X_before_sents = []
        Y = []
        for paraid, scene in paraid2scene.items():
            scene_list = []
            sent_list = []
            before_sent_list = []
            for word in scene:
                word_id = word2id.get(word, 0)
                scene_list.append(word_id)
            for i in range(len(paraid2chars[paraid])):
                char = paraid2chars[paraid][i]
                words = paraid2sents[paraid][i]
                char_id = char2id.get(char, 0)
                char_list = self.onehot_get(char_number, char_id)
                for word in words:
                    word_id_2 = word2id.get(word, 0)
                    sent_list.append(word_id_2)
                X_scene.append(scene_list)
                X_current_sents.append(sent_list)
                X_before_sents.append(before_sent_list)
                Y.append(char_list)
                before_sent_list = sent_list
                sent_list = []
        X_current_sents = pad_sequences(X_current_sents, maxlen=32)
        X_before_sents = pad_sequences(X_before_sents, maxlen=32)
        X_scene = pad_sequences(X_scene, maxlen=16)

        return [X_scene, X_before_sents, X_current_sents], Y

    #看前4/5 预测后1/5
    def generate_X_Y_split(self, characters, paraid2scene, paraid2chars, paraid2sents, episodeid2paraid, para_number, word_dict):
        char2id, id2char, word2id, id2word, char_number, word_number, _ = self.encoding_reduction(characters, word_dict)
        X_scene = []
        X_current_sents = []
        X_before_sents = []
        X_scene_test = []
        X_current_sents_test = []
        X_before_sents_test = []
        Y = []
        Y_test = []
        for paraid, scene in paraid2scene.items():
            scene_list = []
            sent_list = []
            before_sent_list = []
            for word in scene:
                word_id = word2id.get(word, 0)
                scene_list.append(word_id)
            for i in range(len(paraid2chars[paraid])):
                char = paraid2chars[paraid][i]
                words = paraid2sents[paraid][i]
                char_id = char2id.get(char, 0)
                char_list = self.onehot_get(char_number, char_id)
                for word in words:
                    word_id_2 = word2id.get(word, 0)
                    sent_list.append(word_id_2)
                if i < len(paraid2chars[paraid]) * 0.8:
                    X_scene.append(scene_list)
                    X_current_sents.append(sent_list)
                    X_before_sents.append(before_sent_list)
                    Y.append(char_list)
                else:
                    X_scene_test.append(scene_list)
                    X_current_sents_test.append(sent_list)
                    X_before_sents_test.append(before_sent_list)
                    Y_test.append(char_list)
                before_sent_list = sent_list
                sent_list = []
        X_current_sents = pad_sequences(X_current_sents, maxlen=32)
        X_before_sents = pad_sequences(X_before_sents, maxlen=32)
        X_scene = pad_sequences(X_scene, maxlen=16)
        X_current_sents_test = pad_sequences(X_current_sents_test, maxlen=32)
        X_before_sents_test = pad_sequences(X_before_sents_test, maxlen=32)
        X_scene_test = pad_sequences(X_scene_test, maxlen=16)

        return [X_scene, X_before_sents, X_current_sents], Y, [X_scene_test, X_before_sents_test, X_current_sents_test], Y_test

    #不使用embedding
    def generate_X_Y_recudechar(self, characters, paraid2scene, paraid2chars, paraid2sents, episodeid2paraid, para_number, word_dict):
        char2id, id2char, word2id, id2word, char_number, word_number = self.encoding_reduction_onlychar(characters, word_dict)
        X_scene = []
        X_current_sents = []
        X_before_sents = []
        Y = []
        for paraid, scene in paraid2scene.items():
            scene_list = []
            sent_list = []
            before_sent_list = []
            for word in scene:
                word_id = word2id.get(word, 0)
                scene_list.append(word_id)
            for i in range(len(paraid2chars[paraid])):
                char = paraid2chars[paraid][i]
                words = paraid2sents[paraid][i]
                char_id = char2id.get(char, 0)
                char_list = self.onehot_get(char_number, char_id)
                for word in words:
                    word_id_2 = word2id.get(word, 0)
                    sent_list.append(word_id_2)
                X_scene.append(scene_list)
                X_current_sents.append(sent_list)
                X_before_sents.append(before_sent_list)
                Y.append(char_list)
                before_sent_list = sent_list
                sent_list = []
        X_current_sents = pad_sequences(X_current_sents, maxlen=32)
        X_before_sents = pad_sequences(X_before_sents, maxlen=32)
        X_scene = pad_sequences(X_scene, maxlen=16)

        return [X_scene, X_before_sents, X_current_sents], Y

    #加入了之前一句话的说话人,SCENE ENCODING改变
    def generate_X_Y_split_beforechar(self, characters, paraid2scene, paraid2chars, paraid2sents, episodeid2paraid, para_number, word_dict):
        char2id, id2char, word2id, id2word, char_number, word_number, _ = self.encoding_reduction(characters, word_dict)
        X_beforechar = []
        X_beforechar_test = []
        X_scene = []
        X_current_sents = []
        X_before_sents = []
        X_beforechar = []
        X_scene_test = []
        X_current_sents_test = []
        X_before_sents_test = []
        Y = []
        Y_test = []
        for paraid, scene in paraid2scene.items():
            scene_list = []
            sent_list = []
            before_sent_list = []
            before_char_list = [np.zeros(char_number),np.zeros(char_number),np.zeros(char_number)]
            scene_list = self.scene_char_encoder(scene,char2id,char_number)
            for i in range(len(paraid2chars[paraid])):
                char = paraid2chars[paraid][i]
                words = paraid2sents[paraid][i]
                char_id = char2id.get(char, 0)
                char_list = self.onehot_get(char_number, char_id)
                before_char_array = np.concatenate(tuple(before_char_list))
                for word in words:
                    word_id_2 = word2id.get(word, 0)
                    sent_list.append(word_id_2)
                if i < len(paraid2chars[paraid]) * 0.8:
                    X_scene.append(scene_list)
                    X_current_sents.append(sent_list)
                    X_before_sents.append(before_sent_list)
                    X_beforechar.append(before_char_array)
                    Y.append(char_list)
                else:
                    X_scene_test.append(scene_list)
                    X_current_sents_test.append(sent_list)
                    X_before_sents_test.append(before_sent_list)
                    X_beforechar_test.append(before_char_array)
                    Y_test.append(char_list)
                before_sent_list = sent_list
                sent_list = []
                window = len(before_char_list)
                for i in range(window-1):
                    before_char_list[i] = before_char_list[i+1]
                before_char_list[window-1] = char_list
        X_beforechar = np.array(X_beforechar)
        X_beforechar_test = np.array(X_beforechar_test)
        X_current_sents = np.array(pad_sequences(X_current_sents, maxlen=32))
        X_before_sents = np.array(pad_sequences(X_before_sents, maxlen=32))
        X_scene = np.array(X_scene)
        X_current_sents_test = np.array(pad_sequences(X_current_sents_test, maxlen=32))
        X_before_sents_test = np.array(pad_sequences(X_before_sents_test, maxlen=32))
        X_scene_test = np.array(X_scene_test)
        Y = np.array(Y)
        Y_test = np.array(Y_test)

        return [X_scene, X_before_sents, X_current_sents, X_beforechar], Y, [X_scene_test, X_before_sents_test, X_current_sents_test, X_beforechar_test], Y_test




    @staticmethod
    def onehot_get(dim, id):
        onehot_vector = np.zeros(dim)
        onehot_vector[id] = 1
        return onehot_vector

    @staticmethod
    def scene_word_encoder(scene, word2id):
        scene_list = []
        for word in scene:
            word_id = word2id.get(word, 0)
            scene_list.append(word_id)
        return scene_list

    def scene_char_encoder(self,scene, char2id, char_number):
        main_character_list = ['sheldon', 'leonard', 'penny', 'howard', 'raj', 'bernadette', 'amy', 'stuart']
        char_array = np.zeros(4*char_number)
        i = 0
        for char in main_character_list:
            char_normalized = char + '’s'
            if char in scene or char_normalized in scene:
                if i < 4:
                    char_id = char2id.get(char, 0)
                    current_array = self.onehot_get(char_number, char_id)
                    start = i * char_number
                    end = (i + 1) * char_number
                    char_array[start:end] = current_array
                    i += 1
        return char_array











if __name__ == '__main__':
    '''
    preprocessing = Preprocessing()
    characters, paraid2scene, paraid2chars, paraid2sents, episodeid2paraid, para_number, word_dict = preprocessing.load_dataset()
    _, id2char, _, id2word, char_number, word_number= preprocessing.encoding_reduction_onlychar(characters, word_dict)

    X, Y = preprocessing.generate_X_Y_recudechar(characters, paraid2scene, paraid2chars, paraid2sents, episodeid2paraid, para_number, word_dict)


    #print('%s\n%s\n%s\n%s'% (X[0][0],X[1][0],X[2][0],Y[0]))
    #print('%s\n%s\n%s\n%s' % (X[0][1], X[1][1], X[2][1], Y[1]))

    print('%s\n%s\n%s\n%s'% (X[0][40000],X[1][40000],X[2][40000],Y[40000]))
    print('%s\n%s\n%s\n%s' % (X[0][40001], X[1][40001], X[2][40001], Y[40001]))

    '''
    preprocessing = Preprocessing()
    characters, paraid2scene, paraid2chars, paraid2sents, episodeid2paraid, para_number, word_dict = preprocessing.load_dataset()
    preprocessing.feature_read()
    _, id2char, _, id2word, char_number, word_number, embedding_matrix = preprocessing.encoding_reduction(characters, word_dict)
    preprocessing.dataset_count(characters, paraid2scene, paraid2chars, paraid2sents, episodeid2paraid, para_number)
    print(len(characters))
    #X, Y = preprocessing.generate_X_Y(characters, paraid2scene, paraid2chars, paraid2sents, episodeid2paraid, para_number, word_dict)

    X, Y, X_test_2, Y_test_2 = preprocessing.generate_X_Y_split_beforechar(characters, paraid2scene, paraid2chars, paraid2sents, episodeid2paraid,
                                     para_number, word_dict)


    #print('%s\n%s\n%s\n%s'% (X[0][0],X[1][0],X[2][0],Y[0]))
    #print('%s\n%s\n%s\n%s' % (X[0][1], X[1][1], X[2][1], Y[1]))

    #print('%s\n%s\n%s\n%s'% (X[0][0],X[1][0],X[2][0],Y[0]))
    #print('%s\n%s\n%s\n%s' % (X[0][36001], X[1][36001], X[2][36001], Y[36001]))



