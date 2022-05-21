import torch
import numpy as np
from net import Encoder,Decoder,Seq2Seq
import utils
import os
from preprocessing import Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(Data, checkpoint):
    dropout = 0.2
    embed_size = 50
    enc_hidden_size = 100
    dec_hidden_size = 200

    encoder = Encoder(vocab_size=Data.en_total_words,
                    embed_size=embed_size, 
                    enc_hidden_size=enc_hidden_size,
                    dec_hidden_size=dec_hidden_size,
                    directions=2,
                    dropout=dropout)
    decoder = Decoder(vocab_size=Data.cn_total_words,
                    embed_size=embed_size, 
                    enc_hidden_size=enc_hidden_size,
                    dec_hidden_size=dec_hidden_size,
                    dropout=dropout)

    model = Seq2Seq(encoder, decoder)
    model = model.to(device)
    utils.load_checkpoint(checkpoint,model)
    return model

def en2cn_translate(Data,model,sentence_id):
    #英文翻译成中文
    en_sentence = " ".join([Data.inv_en_dict[w] for w in Data.en_datas[sentence_id]]) #英文句子
    cn_sentence = " ".join([Data.inv_cn_dict[w] for w in Data.cn_datas[sentence_id]]) #对应实际的中文句子
    
    batch_x = torch.from_numpy(np.array(Data.en_datas[sentence_id]).reshape(1, -1)).to(device).long()
    batch_x_len = torch.from_numpy(np.array([len(Data.en_datas[sentence_id])])).to(device).long()
    
    #第一个时间步的前项输出
    bos = torch.Tensor([[Data.cn_dict["BOS"]]]).to(device).long()
    
    translation = model.translate(batch_x, batch_x_len, bos, 10)
    translation = [Data.inv_cn_dict[i] for i in translation.data.cpu().numpy().reshape(-1)] #index2word
    
    trans = []
    for word in translation:
        if(word != "EOS"):
            trans.append(word)
        else:
            break
    print(en_sentence)
    print(cn_sentence)
    print(" ".join(trans))

if __name__ == '__main__':
    data = Data()
    restore_path = os.path.join('model', 'epoch_0' + '.pth.tar')
    model = load_model(data,restore_path)

    en2cn_translate(data,model,0)