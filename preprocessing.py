import nltk
import jieba
from collections import Counter
import numpy as np
import os
import json
import logging
dict_path = os.path.join(os.getcwd(),'data')
en_dict_path = os.path.join(dict_path, 'en_dict.json')
cn_dict_path = os.path.join(dict_path, 'cn_dict.json')
en_datas_path = os.path.join(dict_path, 'en_datas.npy')
cn_datas_path = os.path.join(dict_path, 'cn_datas.npy')

class Data:
    def __init__(self, en_file_name="./training-parallel-nc-v12/training/news-commentary-v12.zh-en.en", cn_file_name="./training-parallel-nc-v12/training/news-commentary-v12.zh-en.zh") -> None:
        # 加载数据分词
        if not (os.path.exists(en_dict_path) and os.path.exists(cn_dict_path) and os.path.exists(en_datas_path) and os.path.exists(cn_datas_path)):
            print('加载数据分词...')
            _en = self.load_data(en_file_name, True)
            _cn = self.load_data(cn_file_name, False)
        print('创建词典...')
        # 创建词典 en_dict word2index en_total_words表示单词个数
        if os.path.exists(en_dict_path) and os.path.exists(cn_dict_path):
            with open(en_dict_path, 'r', encoding="utf-8") as f:
                self.en_dict = json.load(f)
                self.en_total_words = len(self.en_dict) + 2
            with open(cn_dict_path, 'r', encoding="utf-8") as f:
                self.cn_dict = json.load(f)
                self.cn_total_words = len(self.cn_dict) + 2
        else:
            self.en_dict, self.en_total_words = self.create_dict(_en)
            self.cn_dict, self.cn_total_words = self.create_dict(_cn)
            with open(en_dict_path, 'w', encoding="utf-8") as f:
                json.dump(self.en_dict, f)
            with open(cn_dict_path, 'w', encoding="utf-8") as f:
                json.dump(self.cn_dict, f, ensure_ascii=False)

        # index2word
        self.inv_en_dict = {v: k for k, v in self.en_dict.items()}
        self.inv_cn_dict = {v: k for k, v in self.cn_dict.items()}
        # 句子编码：将句子中的词转换为词表中的index
        print('句子编码...')
        if os.path.exists(en_datas_path) and os.path.exists(cn_datas_path):
            temp = np.load(en_datas_path,allow_pickle=True)
            self.en_datas = temp.tolist()
            temp= np.load(cn_datas_path,allow_pickle=True)
            self.cn_datas = temp.tolist()
        else:
            self.en_datas, self.cn_datas = self.encode(_en, _cn, self.en_dict, self.cn_dict, sorted_by_len=True)
            np.save(en_datas_path,np.array(self.en_datas))
            np.save(cn_datas_path,np.array(self.cn_datas))

    
    def load_data(self, file_name, is_en):
        # 逐句读取文本，并将句子进行分词，且在句子前面加上'BOS'表示句子开始，在句子末尾加上'EOS'表示句子结束
        datas = []
        with open(file_name, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # if(i>10): # for debug
                #    break
                line = line.strip()
                if(is_en):
                    datas.append(["BOS"] + nltk.word_tokenize(line.lower()) + ["EOS"])
                else:
                    datas.append(["BOS"] + list(jieba.cut(line, cut_all=False)) + ["EOS"])
        return datas


    def create_dict(self, sentences, max_words=50000):
        #统计文本中每个词出现的频数，并用出现次数最多的max_words个词创建词典，
        #且在词典中加入'UNK'表示词典中未出现的词，'PAD'表示后续句子中添加的padding（保证每个batch中的句子等长）
        word_count = Counter()
        for sentence in sentences:
            for word in sentence:
                word_count[word] += 1
        
        most_common_words = word_count.most_common(max_words)  #最常见的max_words个词
        total_words = len(most_common_words) + 2  #总词量（+2：词典中添加了“UNK”和“PAD”）
        word_dict = {w[0]: index+2 for index, w in enumerate(most_common_words)}  #word2index
        word_dict["PAD"] = 0
        word_dict["UNK"] = 1
        return word_dict, total_words

    def encode(self, en_sentences, cn_sentences, en_dict, cn_dict, sorted_by_len):
        #句子编码：将句子中的词转换为词表中的index
        
        #不在词典中的词用”UNK“表示
        out_en_sentences = [[en_dict.get(w, en_dict['UNK']) for w in sentence] for sentence in en_sentences]
        out_cn_sentences = [[cn_dict.get(w, cn_dict['UNK']) for w in sentence] for sentence in cn_sentences]
        
        #基于英文句子的长度进行排序，返回排序后句子在原始文本中的下标
        #目的：为使每个batch中的句子等长时，需要加padding；长度相近的放入一个batch，可使得添加的padding更少
        if(sorted_by_len):
            sorted_index = sorted(range(len(out_en_sentences)), key=lambda idx: len(out_en_sentences[idx]))
            out_en_sentences = [out_en_sentences[i] for i in sorted_index]
            out_cn_sentences = [out_cn_sentences[i] for i in sorted_index]
            
            # 将长度为2的英文删掉(因为有些中文对应的是空白)
            count = 0
            while len(out_en_sentences[0]) <= 2:
                out_en_sentences.pop(0)
                out_cn_sentences.pop(0)
                count += 1
            print(f'一共删除了{count}个句子')
        return out_en_sentences, out_cn_sentences


class Dataset():
    def __init__(self,en, cn, batch_size=8):
        self.dataset = self.generate_dataset(en, cn, batch_size)

    def get_batches(self, num_sentences, batch_size, shuffle=True):
        #用每个句子在原始文本中的（位置）行号创建每个batch的数据索引
        batch_first_idx = np.arange(start=0, stop=num_sentences, step=batch_size) #每个batch中第一个句子在文本中的位置（行号）
        if(shuffle):
            np.random.shuffle(batch_first_idx)
        
        batches = []
        for first_idx in batch_first_idx:
            batch = np.arange(first_idx, min(first_idx+batch_size, num_sentences), 1) #每个batch中句子的位置（行号）
            batches.append(batch)
        return batches

    def add_padding(self, batch_sentences):
        #为每个batch的数据添加padding，并记录下句子原本的长度
        lengths = [len(sentence) for sentence in batch_sentences] #每个句子的实际长度
        max_len = np.max(lengths) #当前batch中最长句子的长度
        data = []
        for sentence in batch_sentences:
            sen_len = len(sentence)
            #将每个句子末尾添0，使得每个batch中的句子等长（后续将每个batch数据转换成tensor时，每个batch中的数据维度必须一致）
            sentence = sentence + [0]*(max_len - sen_len) 
            data.append(sentence)
        data = np.array(data).astype('int32')
        data_lengths = np.array(lengths).astype('int32')
        return data, data_lengths

    def generate_dataset(self, en, cn, batch_size):
        #生成数据集
        batches = self.get_batches(len(en), batch_size)
        datasets = []
        for batch in batches:
            batch_en = [en[idx] for idx in batch]
            batch_cn = [cn[idx] for idx in batch]
            batch_x, batch_x_len = self.add_padding(batch_en)
            batch_y, batch_y_len = self.add_padding(batch_cn)
            datasets.append((batch_x, batch_x_len, batch_y, batch_y_len))
        return datasets

if __name__ == '__main__':
    en_path = "./training-parallel-nc-v12/training/news-commentary-v12.zh-en.en"
    cn_path = "./training-parallel-nc-v12/training/news-commentary-v12.zh-en.zh"

    D = Data(en_path,cn_path)  # 这里面存了词典，存了数据集
    
    dataset = Dataset(D.en_datas,D.cn_datas).dataset
    # print(dataset[0])