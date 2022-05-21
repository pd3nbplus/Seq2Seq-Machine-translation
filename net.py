import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, directions, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # the input size of gru is [sentence_len, batch_size, word_embedding_size]
        # if batch_first=True  => [batch_size, sentence_len, word_embedding_size]
        self.gru = nn.GRU(embed_size, enc_hidden_size, batch_first=True, bidirectional=(directions == 2))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size*2, dec_hidden_size)
        
    def forward(self, batch_x, lengths):
        # batch_x: [batch_size, max_x_sentence_len(n_steps)] 每个位置的元素是一个单词索引
        # lengths: [batch_size]
        
        # 基于每个batch中句子的实际长度倒序（后续使用pad_packed_sequence要求句子长度需要倒排序）
        sorted_lengths, sorted_index = lengths.sort(0, descending=True) 
        batch_x_sorted = batch_x[sorted_index.long()]
        
        # embed shape = （[batch_size, max_x_sentence_len, embed_size]）
        embed = self.embedding(batch_x_sorted)
        embed = self.dropout(embed)
        
        # 将句子末尾添加的padding去掉，使得GRU只对实际有效语句进行编码，
        # 得到PackedSequence类型的object，可以直接传给RNN
        # （RNN的源码中的forward函数里上来就是判断输入是否是PackedSequence的实例，进而采取不同的操作，如果是则输出也是该类型。）
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed, sorted_lengths.long().cpu().data.numpy(), batch_first=True)
        # packed_out为PackedSequence类型数据，hidden为tensor类型:[num_layers * num_directions, batch_size, enc_hidden_size]
        packed_out, hidden = self.gru(packed_embed) 
        
        #unpacked，恢复数据为tensor
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True) #[batch_size, max_x_sentence_len, enc_hidden_size * 2]
        
        #恢复batch中sentence原始的顺序
        _, original_index = sorted_index.sort(0, descending=False)
        out = out[original_index.long()].contiguous()
        hidden = hidden[:, original_index.long()].contiguous()
        
        # 这里写的不够灵活，最终是设置了bidirectional 所以是2
        hidden = torch.cat((hidden[0], hidden[1]), dim=1) #[batch_size, enc_hidden_size*2]
        
        hidden = torch.tanh(self.fc(hidden)).unsqueeze(0)  #[1, batch_size, dec_hidden_size]
        
        return out, hidden # [batch_size, max_x_sentence_len, enc_hidden_size*2], [1, batch_size, dec_hidden_size]


class Attention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(Attention, self).__init__()
        self.enc_hidden_size = encoder_hidden_size
        self.dec_hidden_size = decoder_hidden_size
        
        # 这里计算相似度矩阵的方法是 kq矩阵相乘法
        # k_i = W_K * h_i (for i to max_x_sentence_len)
        # q_0 = W_Q * s_0  但是这里没有矩阵W_Q
        # a_i = k_i^T * q_0 (for i to max_x_sentence_len)
        self.linear_in = nn.Linear(encoder_hidden_size*2, decoder_hidden_size, bias=False)
        self.linear_out = nn.Linear(encoder_hidden_size*2 + decoder_hidden_size, decoder_hidden_size)
        
    def forward(self, output, context, masks):
        # output [batch_size, max_y_sentence_len, dec_hidden_size]
        # (encoder的output)context [batch_size, max_x_sentence_len, enc_hidden_size*2]
        # masks [batch_size, max_y_sentence_len, max_x_sentence_len]
        
        batch_size = output.size(0)
        y_len = output.size(1)
        x_len = context.size(1)
        
        # x shape [batch_size * max_x_sentence_len, enc_hidden_size*2]
        x = context.view(batch_size*x_len, -1) 
        # x shape [batch_size * max_x_len, dec_hidden_size]
        x = self.linear_in(x) 

        # context_in shape [batch_size, max_x_sentence_len, dec_hidden_size]
        context_in = x.view(batch_size, x_len, -1) 
        # bmm (batch matrix mutal) 第一维是batch_size不变，剩下两维度做矩阵相乘
        # atten shape [batch_size, max_y_sentence_len, max_x_sentence_len]
        atten = torch.bmm(output, context_in.transpose(1,2))
        
        # masked_fill方法有两个参数，maske和value，mask是一个pytorch张量（Tensor），
        # 元素是布尔值，value是要填充的值，填充规则是mask中取值为True位置对应于self的相应位置用value填充。
        atten.data.masked_fill_(masks.bool(), -1e-6)
        
        # [batch_size, max_y_sentence_len, max_x_sentence_len]
        atten = F.softmax(atten, dim=2)
        
        # 目标语言上一个词(因为目标语言做了shift处理，所以是上一个词)的编码与源语言所有词经attention的加权，concat上目标语言上一个词的编码，目标语言当前词的预测编码
        # 将atten权重与Encoder的每个隐状态相乘相加(就是矩阵相乘) context(就是那个C，总共有max_y_sentence_len个C)
        # shape = ([batch_size, max_y_sentence_len, enc_hidden_size*2])
        context = torch.bmm(atten, context)
        # [batch_size, max_y_sentence_len, enc_hidden_size*2+dec_hidden_size]
        output = torch.cat((context, output), dim=2)
        
        output = output.view(batch_size*y_len, -1) #[batch_size * max_y_sentence_len, enc_hidden_size*2+dec_hidden_size]
        output = torch.tanh(self.linear_out(output))
        
        output = output.view(batch_size, y_len, -1) # [batch_size, max_y_sentence_len, dec_hidden_size]
        
        return output, atten # [batch_size, max_y_sentence_len, dec_hidden_size]， [batch_size, max_y_sentence_len, max_x_sentence_len]


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.gru = nn.GRU(embed_size, dec_hidden_size, batch_first=True)
        # 将每个输出都映射会词表维度，最大值所在的位置对应的词就是预测的目标词
        self.liner = nn.Linear(dec_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def create_atten_masks(self, x_len, y_len):
        # 创建attention的masks
        # 超出句子有效长度部分的attention用一个很小的数填充，使其在softmax后的权重很小
        max_x_len = x_len.max()
        max_y_len = y_len.max()
        x_masks = torch.arange(max_x_len, device=device)[None,:] < x_len[:, None] #[batch_size, max_x_sentence_len]
        y_masks = torch.arange(max_y_len, device=device)[None,:] < y_len[:, None] #[batch_size, max_y_sentence_len]
        
        #x_masks[:, :, None] [batch_size, max_x_sentence_len, 1]
        #y_masks[:, None, :][batch_size, 1, max_y_sentence_len]
        #masked_fill_填充的是True所在的维度，所以取反(~)
        masks = (~(y_masks[:, :, None] * x_masks[:, None, :])).byte()  #[batch_size, max_y_sentence_len, max_x_sentence_len]
        
        return masks   #[batch_size, max_y_sentence_len, max_x_sentence_len]
    
    def forward(self, encoder_out, x_lengths, batch_y, y_lengths, encoder_hidden):
        # encoder_out ： [batch_size, max_x_sentence_len, enc_hidden_size*2]
        # batch_y: [batch_size, max_y_setence_len]
        # lengths: [batch_size]
        # encoder_hidden: [1, batch_size, dec_hidden_size]
        
        #基于每个batch中句子的实际长度倒序
        sorted_lengths, sorted_index = y_lengths.sort(0, descending=True) 
        batch_y_sorted = batch_y[sorted_index.long()]
        hidden = encoder_hidden[:, sorted_index.long()]
        
        # [batch_size, max_y_setence_len, embed_size]
        embed = self.embedding(batch_y_sorted) 
        embed = self.dropout(embed)
        
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed, sorted_lengths.long().cpu().data.numpy(), batch_first=True)
        # 目标语言编码，h0为编码器中最后一个unit输出的hidden
        packed_out, hidden = self.gru(packed_embed, hidden)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        _, original_index = sorted_index.sort(0, descending=False)
        # [batch_size, max_y_sentence_len, dec_hidden_size]
        out = out[original_index.long()].contiguous()
        # [1, batch_size, dec_hidden_size]
        hidden = hidden[:, original_index.long()].contiguous()
        
        # [batch_size, max_y_sentcnec_len, max_x_sentcnec_len]
        atten_masks = self.create_atten_masks(x_lengths, y_lengths)
        # out [batch_size, max_y_sentence_len, dec_hidden_size]
        # atten [batch_size, max_y_sentence_len, max_x_sentence_len]
        out, atten = self.attention(out, encoder_out, atten_masks) 
        
        # [batch_size, max_y_sentence_len, vocab_size]
        out = self.liner(out) 
        
        #log_softmax求出每个输出的概率分布，最大概率出现的位置就是预测的词在词表中的位置
        out = F.log_softmax(out, dim=-1) #[batch_size, max_y_sentence_len, vocab_size]
        return out, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, encoder_hid = self.encoder(x, x_lengths)  # 源语言编码
        output, hidden = self.decoder(encoder_out, x_lengths, y, y_lengths, encoder_hid) # 解码出目标
        return output
    
    def translate(self, x, x_lengths, y, max_length=50):
        # 翻译en2cn
        # max_length表示翻译的目标句子可能的最大长度
        encoder_out , encoder_hidden = self.encoder(x, x_lengths) #将输入的英文进行编码
        predicts = []
        batch_size = x.size(0)
        #目标语言（中文）的输入只有”BOS“表示句子开始，因此y的长度为1
        #每次都用上一个词(y)与编码器的输出预测下一个词，因此y的长度一直为1
        y_length = torch.ones(batch_size).long().to(y.device)
        for i in range(max_length):
            #每次用上一次的输出y和编码器的输出encoder_hidden预测下一个词
            output, hidden = self.decoder(encoder_out, x_lengths, y, y_length, encoder_hidden)
            #output: [batch_size, 1, vocab_size]
            
            #output.max(2)[1]表示找出output第二个维度的最大值所在的位置（即预测词在词典中的index）
            y = output.max(2)[1].view(batch_size, 1) #[batch_size, 1]
            predicts.append(y)
            
        predicts = torch.cat(predicts, 1) #[batch_size, max_length]
       
        return predicts

#自定义损失函数
#目的：使句子中添加的padding部分不参与损失计算
class MaskCriterion(nn.Module):
    def __init__(self):
        super(MaskCriterion, self).__init__()
        
    def forward(self, predicts, targets, masks):
        #predicts [batch_size, max_y_sentence_len, vocab_size]
        #target [batch_size, max_y_sentence_len]
        #masks [batch_size, max_y_sentence_len]
        
        predicts = predicts.contiguous().view(-1, predicts.size(2))  #[batch_size * max_y_sentence_len, vocab_size]
        targets = targets.contiguous().view(-1, 1)   #[batch_size*max_y_sentence_len, 1]
        masks = masks.contiguous().view(-1, 1)   #[batch_size*max_y_sentence_len, 1]
        
        # predicts.gather(1, targets)为predicts[i][targets[i]]
        # 乘上masks，即只需要计算句子有效长度的预测
        # 负号：因为采用梯度下降法，所以要最大化目标词语的概率，即最小化其相反数
        loss = -predicts.gather(1, targets) * masks
        loss = torch.sum(loss) / torch.sum(masks) #平均
        
        return loss