import torch

from Utils import truncate_pad, AverageMeter
from Utils.dl_utils import BLEU


def seq2seq_predict(model, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """Predict for sequence to sequence.

    Defined in :numref:`sec_seq2seq_training`"""
    # Set `net` to eval mode for inference
    model.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])

    # Add the batch axis
    enc_X = torch.tensor(src_tokens, dtype=torch.long, device=device).unsqueeze(0)

    enc_outputs = model.encoder(enc_X, enc_valid_len)

    dec_state = model.decoder.init_state(enc_outputs, enc_valid_len)

    # Add the batch axis
    dec_X = torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device).unsqueeze(0)
    output_seq, attention_weight_seq = [], []

    for _ in range(num_steps):
        Y, dec_state = model.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(model.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    translation = ' '.join(tgt_vocab.to_tokens(output_seq))
    return translation, attention_weight_seq

def seq2seq_test(model, test_loader, src_vocab, tgt_vocab, num_steps,
                 device, save_attention_weights=False):
    bleu_meter = AverageMeter()
    translations, attention_weight_seqs = [], []
    src_list, tgt_list = [], []
    with torch.no_grad():
        for srcs, tgts in test_loader:
            for i in range(len(srcs)):
                len_label = len(tgts[i].split(' '))
                if len_label > 1:
                    translation, attention_weight_seq = \
                        seq2seq_predict(model, srcs[i], src_vocab, tgt_vocab,
                                        num_steps, device, save_attention_weights)
                    translations.append(translation)
                    attention_weight_seqs.append(attention_weight_seq)
                    src_list.append(srcs[i])
                    tgt_list.append(tgts[i])
                    bleu = BLEU(translation, tgts[i], 2)
                    bleu_meter.update(bleu, 1)
    return bleu_meter.avg, (src_list, tgt_list, translations), attention_weight_seqs
