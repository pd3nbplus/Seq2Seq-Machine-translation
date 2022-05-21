from net import Encoder, Decoder, Seq2Seq, MaskCriterion
import torch, gc
from preprocessing import Data,Dataset
import utils
import os
import logging
from tqdm import tqdm

logger = logging.getLogger('Attention')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, data, epoches,restore_file='attention'):
    restore_path = os.path.join('model', restore_file + '.pth.tar')
    if restore_file is not None and os.path.exists(restore_path):
        logger.info('Restoring parameters from {}'.format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
    logger.info('begin training and evaluation')
    test_datasets = []
    for epoch in range(epoches):
        logger.info('Epoch {}/{}'.format(epoch + 1, epoches))
        model.train()
        total_words = 0
        total_loss = 0.
        for it, (batch_x, batch_x_len, batch_y, batch_y_len) in enumerate(tqdm(data)):
            torch.cuda.empty_cache()
            #创建验证数据集
            if(epoch == 0 and it % 10 == 0):
                test_datasets.append((batch_x, batch_x_len, batch_y, batch_y_len))
                continue
            batch_x = torch.from_numpy(batch_x).to(device).long()
            batch_x_len = torch.from_numpy(batch_x_len).to(device).long()
            
            #因为训练（或验证）时，decoder根据上一步的输出（预测词）和encoder_out经attention的加权和，以及上一步输出对应的实际词预测下一个词
            #所以输入到decoder中的目标语句为[BOS, word_1, word_2, ..., word_n]
            #预测的实际标签为[word_1, word_2, ..., word_n, EOS]
            batch_y_decoder_input = torch.from_numpy(batch_y[:, :-1]).to(device).long()
            batch_targets = torch.from_numpy(batch_y[:, 1:]).to(device).long()
            batch_y_len = torch.from_numpy(batch_y_len-1).to(device).long()
            batch_y_len[batch_y_len<=0] = 1
            
            batch_predicts = model(batch_x, batch_x_len, batch_y_decoder_input, batch_y_len)
            
            #生成masks：
            batch_y_len = batch_y_len.unsqueeze(1) #[batch_size, 1]
            batch_target_masks = torch.arange(batch_y_len.max().item(), device=device) < batch_y_len
            batch_target_masks = batch_target_masks.float()
            batch_y_len = batch_y_len.squeeze(1) #[batch_size]
            
            loss = loss_func(batch_predicts, batch_targets, batch_target_masks)
            
            num_words = torch.sum(batch_y_len).item() #每个batch总的词量
            total_loss += loss.item() * num_words
            total_words += num_words
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            
            # if(it % 5 == 0):
            #     logger.info("Epoch {} / {}, Iteration: {}, Train Loss: {}".format(epoch, epoches, it, loss.item()))
        logger.info("Epoch {} / {}, Train Loss: {}".format(epoch+1, epoches, total_loss/total_words))
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              epoch=epoch,
                              is_best=False,
                              checkpoint='model')
        if(epoch!=0 and epoch % 100 == 0):
            test(model, test_datasets)

def test(mode, data):
    model.eval()
    total_words = 0
    total_loss = 0.
    with torch.no_grad():
        for i, (batch_x, batch_x_len, batch_y, batch_y_len) in enumerate(data):
            batch_x = torch.from_numpy(batch_x).to(device).long() 
            batch_x_len = torch.from_numpy(batch_x_len).to(device).long()
            
            batch_y_decoder_input = torch.from_numpy(batch_y[:, :-1]).to(device).long()
            batch_targets = torch.from_numpy(batch_y[:, 1:]).to(device).long()
            batch_y_len = torch.from_numpy(batch_y_len-1).to(device).long()
            batch_y_len[batch_y_len<=0] = 1
            
            batch_predicts = model(batch_x, batch_x_len, batch_y_decoder_input, batch_y_len)
            
            batch_target_masks = torch.arange(batch_y_len.max().item(), device=device)[None, :] < batch_y_len[:, None]
            batch_target_masks = batch_target_masks.float()
            
            loss = loss_func(batch_predicts, batch_targets, batch_target_masks)
            
            num_words = torch.sum(batch_y_len).item()
            total_loss += loss.item() * num_words
            total_words += num_words
        print("Test Loss:", total_loss/total_words)


if __name__ == '__main__':
    en_path = "./training-parallel-nc-v12/training/news-commentary-v12.zh-en.en"
    cn_path = "./training-parallel-nc-v12/training/news-commentary-v12.zh-en.zh"

    utils.set_logger(os.path.join(os.path.join(os.getcwd(),'model'), 'train.log'))
    logger.info(f'{device}')

    logger.info('Loading the datasets...')
    D = Data(en_path,cn_path)
    datasets = Dataset(D.en_datas,D.cn_datas,4).dataset
    logger.info('Loading complete.')
    
    dropout = 0.2
    embed_size = 50
    enc_hidden_size = 100
    dec_hidden_size = 200
    num_epochs = 20
    lr = 1e-3

    encoder = Encoder(vocab_size=D.en_total_words,
                    embed_size=embed_size, 
                    enc_hidden_size=enc_hidden_size,
                    dec_hidden_size=dec_hidden_size,
                    directions=2,
                    dropout=dropout)
    decoder = Decoder(vocab_size=D.cn_total_words,
                    embed_size=embed_size, 
                    enc_hidden_size=enc_hidden_size,
                    dec_hidden_size=dec_hidden_size,
                    dropout=dropout)

    model = Seq2Seq(encoder, decoder)
    model = model.to(device)

    logger.info(f'Model: \n{str(model)}')
    loss_func = MaskCriterion().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info('Starting training for {} epoch(s)'.format(num_epochs))
    train(model, datasets, epoches=num_epochs,restore_file='epoch_0')