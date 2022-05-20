import torch
from torch import nn

class ImageEmbedding(nn.Module):
    '''
      对图像进行编码，把图片当做一个句子，把图片分割成块，每一块表示一个单词
    '''

    def __init__(self, patch_size, in_channels=3, hidden_size=32):
        super(ImageEmbedding, self).__init__()
        ##将图片分割成多少块（img_size / patch_size）*（img_size / patch_size）
        # 对图片进行卷积获取图片的块，并且将每一块映射成hidden_size维
        self.patch_embeddings = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_size,
            kernel_size=patch_size, stride=patch_size)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        bsz = x.shape[0]
        x = self.patch_embeddings(x)
        # bs, num_step, hidden_size
        x = x.flatten(2).transpose(-1, -2)
        embeddings = self.dropout(x)
        return embeddings, torch.tensor([embeddings.shape[1] for _ in range(bsz)], device=x.device)