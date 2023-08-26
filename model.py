import torch.nn as nn
import torch
import math


class Attention(nn.Module):
    def __init__(self, libSpectrumIfShareDim: int, attentionDim: int):
        super(Attention, self).__init__()
        self.attentionDim = attentionDim
        # batch * 6 * 1
        self.softmax = nn.Softmax(dim=2)
        self.Q = nn.Linear(libSpectrumIfShareDim, self.attentionDim)
        self.K = nn.Linear(libSpectrumIfShareDim, self.attentionDim)
        self.V = nn.Linear(libSpectrumIfShareDim, self.attentionDim)

    def forward(self, libSpectrumIfShare: torch.Tensor):
        libSpectrumIfShare = libSpectrumIfShare.to(torch.float32)
        q = self.Q(libSpectrumIfShare)
        k = self.K(libSpectrumIfShare)
        v = self.V(libSpectrumIfShare)
        return (
            self.softmax(q @ torch.transpose(k, 1, 2) /
                         math.sqrt(self.attentionDim))
            @ v
        )


class AttentionWeightMatrix(nn.Module):
    def __init__(self, libSpectrumIfShareDim: int, attentionDim: int):
        super(AttentionWeightMatrix, self).__init__()
        self.attentionDim = attentionDim
        self.Q = nn.Linear(libSpectrumIfShareDim, attentionDim)
        self.K = nn.Linear(libSpectrumIfShareDim, attentionDim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, libSpectrumIfShare: torch.Tensor) -> torch.Tensor:
        q = self.Q(libSpectrumIfShare)
        k = self.K(libSpectrumIfShare)
        return torch.transpose(
            self.softmax(q @ torch.transpose(k, 1, 2) /
                         math.sqrt(self.attentionDim)),
            1,
            2,
        )


# 肽段定性模型
class identifyModel(nn.Module):
    def __init__(self, libMzNum: int, libMsMinNum: int, attentionDim: int):
        r"""
        Input:
        * libMzNum 肽段峰数
        * libMsMinNum 肽段匹配的最相关的图谱数
        * attentionDim 注意力机制模块维度
        """
        super(identifyModel, self).__init__()
        self.FeatureTrans = Attention(1, attentionDim)

        self.attention1 = nn.Sequential(
            Attention(attentionDim, attentionDim),
        )

        self.norm1 = nn.Sequential(
            nn.BatchNorm1d(6),
        )

        self.FWN1 = nn.Sequential(
            nn.Linear(attentionDim, attentionDim),
            nn.ReLU(),
            nn.Linear(attentionDim, attentionDim),
        )

        self.FWNNorm1 = nn.Sequential(nn.BatchNorm1d(6))

        self.attention2 = nn.Sequential(Attention(attentionDim, attentionDim))

        self.norm2 = nn.Sequential(nn.BatchNorm1d(6))

        self.FWN2 = nn.Sequential(
            nn.Linear(attentionDim, attentionDim),
            nn.ReLU(),
            nn.Linear(attentionDim, attentionDim),
        )

        self.FWNNorm2 = nn.Sequential(nn.BatchNorm1d(6))

        self.attention3 = nn.Sequential(
            Attention(attentionDim, attentionDim),
        )

        self.norm3 = nn.Sequential(nn.BatchNorm1d(6))

        self.FWN3 = nn.Sequential(
            nn.Linear(attentionDim, attentionDim),
            nn.ReLU(),
            nn.Linear(attentionDim, attentionDim),
        )

        self.FWNNorm3 = nn.Sequential(nn.BatchNorm1d(6))

        self.attentionWeightMatrix = AttentionWeightMatrix(
            attentionDim, attentionDim)
        # 卷积层
        self.Conv = nn.Sequential(
            nn.Conv1d(libMsMinNum + 1, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Dropout(0.1),
        )
        # 全连接层
        self.Linear = nn.Sequential(
            nn.Linear(128 * libMzNum, 256),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def CalculateDistanceAndPeakSum(self, libMz: torch.Tensor, libMsMz: torch.Tensor) -> torch.Tensor:
        r"""
        Input:
        * libMz 肽段参考图谱峰信息
        * libMsMz 肽段匹配的最相关的混合图谱的峰信息

        由于 libMz 的是进行了归一化处理, 因此我们事先要对混合图谱进行同样的操作

        * 归一化处理: Lib_{i} = Lib_{i} / \sum_{j = 1}^n Lib_{j}

        之后计算哈密顿距离和峰强度和向量

        1. 哈密顿距离向量:

        Distance_i^{k}[j] = |Lib_i^{j} - MS2_k^{j}|

        这里得到的 $Distance_i^{k}$ 是个向量，长度为 $Lib_i$ 的峰的个数

        其中, j 表示 Lib_i 和 MS2_k 对应的第 j 个峰, Distance_i^{k} 表示 Lib_i 和 MS2_k 的距离向量。

        2. 将哈密顿距离向量进行对应位置相加

        MS2Sum_i^{j} = \sum_{k = 1}^{s} Distance_k^{j}

        其中, s 表示 Lib_i 匹配到的混合二级质谱的个数, j 表示 Lib_i 和 MS2_k 对应的第 j 个峰。

        最后, 将两者进行合并

        Output: return
        * concat(distance, msMzsumVector)
        """
        msMz = libMsMz[:, :, :, 1]
        libMz = libMz[:, :, :, 1]
        msMzsum = torch.sum(msMz, dim=2, keepdim=True)
        msMzsum = msMzsum + 0.0000001
        normMsMz = msMz / msMzsum
        # 哈密顿距离
        distance = torch.abs(libMz - normMsMz)  # [batch, 5, 6]
        # 计算峰强度和
        msMzsumVector = torch.sum(
            normMsMz, dim=1, keepdim=True)  # [batch, 1, 6]
        x = torch.cat((distance, msMzsumVector), dim=1)  # [batch, 6, 6]
        # 一定要转换为 float 32
        # 不然报错
        x = x.to(torch.float32)
        return x

    # 输入见该代码框的 markdown 文件
    def forward(
        self,
        libMz: torch.Tensor,
        libMsMz: torch.Tensor,
        libSpectrumIfShare: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Input:
        * libMz 肽段参考图谱
        * libMsMz 肽段匹配的最相关的 s 个混合二级图谱
        * libSpectrumIfShare 肽段特异峰向量

        OutPut: return
        * libTarget 肽段的标签
        """
        batch = libMz.size(0)
        libSpectrumIfShare = self.FeatureTrans(libSpectrumIfShare)

        libSpectrumIfShare = self.norm1(
            libSpectrumIfShare + self.attention1(libSpectrumIfShare)
        )

        libSpectrumIfShare = self.FWNNorm1(
            libSpectrumIfShare + self.FWN1(libSpectrumIfShare)
        )

        libSpectrumIfShare = self.norm2(
            libSpectrumIfShare + self.attention2(libSpectrumIfShare)
        )

        libSpectrumIfShare = self.FWNNorm2(
            libSpectrumIfShare + self.FWN2(libSpectrumIfShare)
        )

        libSpectrumIfShare = self.norm3(
            libSpectrumIfShare + self.attention3(libSpectrumIfShare)
        )

        libSpectrumIfShare = self.FWNNorm3(
            libSpectrumIfShare + self.FWN3(libSpectrumIfShare)
        )

        WeightMatrix = self.attentionWeightMatrix(libSpectrumIfShare)
        x = self.CalculateDistanceAndPeakSum(libMz, libMsMz)
        x = torch.bmm(x, WeightMatrix)
        x: torch.Tensor = self.Conv(x)
        x = x.view(batch, -1)
        x = self.Linear(x)
        return x
