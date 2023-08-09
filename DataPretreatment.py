import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
import torch
from typing import Tuple, List


class DIADataSet(Dataset):
    def __init__(
        self,
        libMsMz: np.ndarray,
        libMs: np.ndarray,
        libSpectrumIfShare: np.ndarray,
        libTarget: np.ndarray,
    ) -> None:
        r"""
        Input:
        * libMsMz 肽段匹配到的 s 个最相关的混合图谱
        * libMs 肽段理论二级图谱
        * libSpectrumIfShare 肽段特异峰向量
        * libTarget 标签
        """
        super(DIADataSet, self).__init__()
        self.libSpectrumIfShare = libSpectrumIfShare
        self.libMsMz = libMsMz
        self.libMs = libMs
        self.libTarget = libTarget
        self.len = len(self.libTarget)

    def __len__(self) -> int:
        return self.len

    def __getitem__(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r""" """
        return (
            self.libMsMz[index],
            self.libMs[index],
            self.libSpectrumIfShare[index],
            self.libTarget[index],
        )


def SplitTrainAndValDataSet(
    TrainDataSet: DIADataSet,
    TrainDataDecoySet: DIADataSet,
    lengthValDataSet: int,
    lengthValDataDecoySet: int,
) -> Tuple[DIADataSet, DIADataSet, DIADataSet]:
    r"""
    Input:
    * TrainDataSet: 正样本训练数据
    * TrainDataDecoySet 负样本训练数据
    * lengthValDataSet 正样本验证集大小
    * lengthValDataDecoySet 负样本验证集大小
    1. 根据输入的训练数据, 从正样本和负样本中各抽取数条数据作为验证集，每个 epoch 结束前进行 FDR 控制
       选取大于分数阈值最多的模型
    2. 将剩余训练数据中的正样本和负样本进行合并并且进行随机化选取

    Output:
        返回训练集, 验证集, 诱饵验证集
    """
    trainDataSet, valDataSet = random_split(
        TrainDataSet,
        [len(TrainDataSet) - lengthValDataSet, lengthValDataSet],
        generator=torch.Generator().manual_seed(0),
    )
    trainDataDecoySet, valDataDecoySet = random_split(
        TrainDataDecoySet,
        [len(TrainDataDecoySet) - lengthValDataDecoySet, lengthValDataDecoySet],
        generator=torch.Generator().manual_seed(0),
    )
    trainDataSet = ConcatDataset([trainDataSet, trainDataDecoySet])
    return trainDataSet, valDataSet, valDataDecoySet


def GetAllDataLoader(
    batch_size: int,
    TrainDataSet: DIADataSet,
    TrainDataSetDecoy: DIADataSet,
    testDataSet: DIADataSet,
    testDataSetDecoy: DIADataSet,
    lengthValDataSet: int,
    lengthValDataSetDecoy: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    r"""
    把 GetTrainDataLoader 和 GetTestDataLoader 两个函数操作合并在一起
    """
    trainDataSet, valDataSet, valDataDecoySet = SplitTrainAndValDataSet(
        TrainDataSet, TrainDataSetDecoy, lengthValDataSet, lengthValDataSetDecoy
    )
    trainData, valData, valDataDecoy = GetTrainDataLoader(
        batch_size, trainDataSet, valDataSet, valDataDecoySet
    )
    testData, testDataDecoy = GetTestDataLoader(
        batch_size, testDataSet, testDataSetDecoy
    )
    return trainData, valData, valDataDecoy, testData, testDataDecoy


def GetTrainDataLoader(
    batch_size: int,
    trainDataSet: DIADataSet,
    valDataSet: List[DIADataSet],
    valDataDecoySet: List[DIADataSet],
) -> Tuple[DataLoader, List[DataLoader], List[DataLoader]]:
    r"""
    Input:
    * batch_size: batch 大小
    * trainDataSet: 训练集
    * valDataSet: 验证集
    * valDataDecoySet: 诱饵验证集

    OutPut: return
    * 训练集
    * 正样本验证集
    * 诱饵验证集的 dataloader
    """
    trainData = DataLoader(
        dataset=trainDataSet, shuffle=True, batch_size=batch_size, num_workers=8
    )
    valData = [
        DataLoader(dataset=dataset, shuffle=False,
                   batch_size=batch_size, num_workers=8)
        for dataset in valDataSet
    ]
    valDataDecoy = [
        DataLoader(dataset=dataset, shuffle=False,
                   batch_size=batch_size, num_workers=8)
        for dataset in valDataDecoySet
    ]
    return trainData, valData, valDataDecoy


def GetTestDataLoader(
    batch_size: int, testDataSet: DIADataSet, testDataDecoySet: DIADataSet
) -> Tuple[DataLoader, DataLoader]:
    r"""
    Input:
    * batch_size: batch 大小
    * testDataSet: 测试集
    * testDataDecoySet: 诱饵测试集

    Output: return
    * 测试集
    * 诱饵测试集的 dataloader
    """
    testData = DataLoader(
        dataset=testDataSet, shuffle=False, batch_size=batch_size, num_workers=8
    )
    testDataDecoy = DataLoader(
        dataset=testDataDecoySet, shuffle=False, batch_size=batch_size, num_workers=8
    )
    return testData, testDataDecoy


def LibSpectrumIfSharePreTreatMent(LibSpectrumIfShare: np.ndarray) -> np.ndarray:
    r"""
    对特异峰向量进行预处理操作
    [0, 0, 0, 0, 1, 1] -> [[0], [0], [0], [0], [1], [1]]
    """
    x = LibSpectrumIfShare
    y = []
    for item in x:
        y.append(item)
    testLibSpectrumIfShare = np.array(y)
    testLibSpectrumIfShare = np.reshape(
        testLibSpectrumIfShare, (np.shape(testLibSpectrumIfShare)[0], 6, 1)
    )
    return testLibSpectrumIfShare


def LibMsMzPreTreatMent(LibMsMz: np.ndarray) -> np.ndarray:
    r"""
    1. 匹配到的谱图进行预处理操作
    2. 归一化等或不进行操作
    """
    Ms = []
    for data in LibMsMz:
        Ms.append(data)
    test = np.array(Ms)
    msIntensity = test[:, :, :, [1]]
    msIntensity = msIntensity
    msMz = test[:, :, :, [0]]
    Ms = np.concatenate((msMz, msIntensity), axis=3)
    LibMsMz = Ms
    return LibMsMz


def Pretreatment(
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    对图谱和特异峰向量进行预处理操作
    """
    LibMsMz, LibMz, LibTarget, LibSpectrumIfShare, fileWinLib = (
        data[0],
        data[1],
        data[2],
        data[3],
        data[4],
    )
    LibSpectrumIfShare = LibSpectrumIfSharePreTreatMent(LibSpectrumIfShare)
    LibMsMz = LibMsMzPreTreatMent(LibMsMz)
    return LibMsMz, LibMz, LibTarget, LibSpectrumIfShare, fileWinLib


def GetTrainDataSet(data: np.ndarray) -> DIADataSet:
    r"""
    Input:
    * data 输入的初始训练数据

    事先对肽段匹配上的图谱、肽段参考图谱、肽段特异峰向量进行预处理操作, 详情见 Pretreatment 函数

    Output: return
    * 处理之后的数据集
    """
    LibMsMz, LibMz, LibTarget, LibSpectrumIfShare, _ = Pretreatment(data)
    return DIADataSet(LibMsMz, LibMz, LibSpectrumIfShare, LibTarget)


def GetTestDataSetAndFilewindow(
    data: np.ndarray,
) -> Tuple[DIADataSet, Tuple[str, str, Tuple[str, int]]]:
    r"""
    Input:
    * data 输入的初始测试数据

    事先对肽段匹配上的图谱、肽段参考图谱、肽段特异峰向量进行预处理操作, 详情见 Pretreatment 函数

    Output: return
    * 处理之后的数据集
    * (文件名、窗口信息、肽段名)
    """
    LibMsMz, LibMz, LibTarget, LibSpectrumIfShare, FileWinLib = Pretreatment(
        data)
    return DIADataSet(LibMsMz, LibMz, LibSpectrumIfShare, LibTarget), FileWinLib


def GetTestData(
    batch_size: int, testPath: str, testDecoyPath: str
) -> Tuple[DataLoader, DataLoader, Tuple[str], Tuple[str]]:
    r"""
    Input:
    * batch_size batch 大小
    * testPath 测试数据路径
    * testDecoyPath 测试诱饵数据路径

    Output: return
    * 测试数据 DataLoader
    * 测试诱饵数据 DataLoader
    * 测试数据 (文件, 窗口, (肽段名, 带电荷数))
    * 测试诱饵数据 (文件, 窗口, (肽段名, 带电荷数))
    """
    TestData = np.load(testPath, allow_pickle=True)
    TestDataDecoy = np.load(testDecoyPath, allow_pickle=True)

    testDataSet, fileWinLib = GetTestDataSetAndFilewindow(TestData)
    testDataDecoySet, fileWinLibDecoy = GetTestDataSetAndFilewindow(
        TestDataDecoy)

    testData, testDataDecoy = GetTestDataLoader(
        batch_size, testDataSet, testDataDecoySet
    )

    return testData, testDataDecoy, fileWinLib, fileWinLibDecoy


def GetLFQAndTrypsin_HFXData(
    batch_size: int,
    lfqPath: str,
    lfqDecoyPath: str,
    Trypsin_HFXpath: str,
    Trypsin_HFXdecoyPath: str,
) -> Tuple[DataLoader, List[DataLoader], List[DataLoader]]:
    r"""
    Input:
    * batch_size: batch 大小
    * lfqPath: lfq 正样本训练数据路径  (可能为 lfq32 和 lfq64)
    * lfqDecoyPath: lfq 负样本训练数据路径 (可能为 lfq32 和 lfq64)
    * Trypsin_HFXPath: 血浆正样本训练数据路径
    * Trypsin_HFXDecoyPath: 血浆负样本训练数据路径

    获取 lfq 和 trupsin_HFX 血浆的训练数据

    Output: return
    * 训练集
    * 正样本验证集
    * 诱饵验证集的 dataloader
    """
    # lfq 和血浆数据
    lfqTrainData = np.load(lfqPath, allow_pickle=True)
    lfqDecoyData = np.load(lfqDecoyPath, allow_pickle=True)
    Trypsin_HFXTrainData = np.load(Trypsin_HFXpath, allow_pickle=True)
    Trypsin_HFXDecoyData = np.load(Trypsin_HFXdecoyPath, allow_pickle=True)
    # 获取数据的 DataSet
    lfqTrainDataSet = GetTrainDataSet(lfqTrainData)
    lfqTrainDataDecoySet = GetTrainDataSet(lfqDecoyData)
    Trypsin_HFXTrainDataSet = GetTrainDataSet(Trypsin_HFXTrainData)
    Trypsin_HFXTrainDataDecoySet = GetTrainDataSet(Trypsin_HFXDecoyData)

    """
    由于血浆和 lfq 样本数量不一致, 有训练集和验证集拆分可以采用两种不同的方法
    一、
        实现将两者进行合并处理, 再从正样本和负样本中各抽取 2w 条数据作为验证集和诱饵验证集

    二、
    1. 血浆从正样本和负样本中各抽取 5k 条作为验证集和诱饵验证集, lfq 从正样本和负样本中抽取 1w 条作为验证集和诱饵验证集
    2. 把两者训练集、验证集、诱饵验证集进行合并, 训练集随机化
    """
    # TrainDataSet = ConcatDataset([lfqTrainDataSet, Trypsin_HFXTrainDataSet])
    # TrainDataDecoySet = ConcatDataset(
    #     [lfqTrainDataDecoySet, Trypsin_HFXTrainDataDecoySet]
    # )

    # trainDataSet, valDataSet, valDataDecoySet = SplitTrainAndValDataSet(
    #     TrainDataSet, TrainDataDecoySet, 20000, 20000
    # )

    lfqtrainDataSet, lfqvalDataSet, lfqvalDataDecoySet = SplitTrainAndValDataSet(
        lfqTrainDataSet, lfqTrainDataDecoySet, 10000, 10000
    )

    (
        Trypsin_HFXtrainDataSet,
        Trypsin_HFXvalDataSet,
        Trypsin_HFXvalDataDecoySet,
    ) = SplitTrainAndValDataSet(
        Trypsin_HFXTrainDataSet, Trypsin_HFXTrainDataDecoySet, 5000, 5000
    )

    trainDataSet = ConcatDataset([lfqtrainDataSet, Trypsin_HFXtrainDataSet])
    # ConcatDataset([lfqvalDataSet, Trypsin_HFXvalDataSet]),
    # ConcatDataset([lfqvalDataDecoySet, Trypsin_HFXvalDataDecoySet]),

    valDataSet, valDataDecoySet = [lfqvalDataSet, Trypsin_HFXvalDataSet], [
        lfqvalDataDecoySet,
        Trypsin_HFXvalDataDecoySet,
    ]

    trainData, valData, valDataDecoy = GetTrainDataLoader(
        batch_size, trainDataSet, valDataSet, valDataDecoySet
    )
    return trainData, valData, valDataDecoy


def GetAllTrainData(batch_size: int,
                    lfq32Path: str,
                    lfq32DecoyPath: str,
                    lfq64Path: str,
                    lfq64DecoyPath: str,
                    tims20211002Path: str,
                    tims20211002DecoyPath: str,
                    Trypsin_HFXpath: str,
                    Trypsin_HFXdecoyPath: str) -> Tuple[DataLoader, List[DataLoader], List[DataLoader]]:
    r"""
    Input:
    * batch_size: batch 大小
    * lfq32Path: lfq32 正样本训练数据路径
    * lfq32DecoyPath: lfq32 负样本训练数据路径
    * lfq64Path: lfq64 正样本训练数据路径
    * lfq64DecoyPath: lfq64 负样本训练数据路径
    * tims20211002Path: tims 血浆正样本训练数据路径
    * tims20211002Path: tims 血浆负样本训练数据路径
    * Trypsin_HFXPath: Trypsin_HFX 血浆正样本训练数据路径
    * Trypsin_HFXDecoyPath: Trypsin_HFX 血浆负样本训练数据路径

    获取 lfq 和两类血浆的训练数据

    Output: return
    * 训练集
    * 正样本验证集
    * 诱饵验证集的 dataloader
    """
    # lfq 和血浆数据
    lfq32TrainData = np.load(lfq32Path, allow_pickle=True)
    lfq32DecoyData = np.load(lfq32DecoyPath, allow_pickle=True)
    lfq64TrainData = np.load(lfq64Path, allow_pickle=True)
    lfq64DecoyData = np.load(lfq64DecoyPath, allow_pickle=True)
    tims20211002TrainData = np.load(tims20211002Path, allow_pickle=True)
    tims20211002DecoyData = np.load(tims20211002DecoyPath, allow_pickle=True)
    Trypsin_HFXTrainData = np.load(Trypsin_HFXpath, allow_pickle=True)
    Trypsin_HFXDecoyData = np.load(Trypsin_HFXdecoyPath, allow_pickle=True)
    # 获取数据的 DataSet
    lfq32TrainDataSet = GetTrainDataSet(lfq32TrainData)
    lfq32TrainDataDecoySet = GetTrainDataSet(lfq32DecoyData)
    lfq64TrainDataSet = GetTrainDataSet(lfq64TrainData)
    lfq64TrainDataDecoySet = GetTrainDataSet(lfq64DecoyData)
    tims20211002TrainDataSet = GetTrainDataSet(tims20211002TrainData)
    tims20211002TrainDataDecoySet = GetTrainDataSet(tims20211002DecoyData)
    Trypsin_HFXTrainDataSet = GetTrainDataSet(Trypsin_HFXTrainData)
    Trypsin_HFXTrainDataDecoySet = GetTrainDataSet(Trypsin_HFXDecoyData)

    lfq32trainDataSet, lfq32valDataSet, lfq32valDataDecoySet = SplitTrainAndValDataSet(
        lfq32TrainDataSet, lfq32TrainDataDecoySet, 10000, 10000
    )

    lfq64trainDataSet, lfq64valDataSet, lfq64valDataDecoySet = SplitTrainAndValDataSet(
        lfq64TrainDataSet, lfq64TrainDataDecoySet, 10000, 10000
    )

    tims20211002trainDataSet, tims20211002valDataSet, tims20211002valDataDecoySet = SplitTrainAndValDataSet(
        tims20211002TrainDataSet, lfq64TrainDataDecoySet, 1000, 1000
    )

    (
        Trypsin_HFXtrainDataSet,
        Trypsin_HFXvalDataSet,
        Trypsin_HFXvalDataDecoySet,
    ) = SplitTrainAndValDataSet(
        Trypsin_HFXTrainDataSet, Trypsin_HFXTrainDataDecoySet, 1000, 1000
    )

    trainDataSet = ConcatDataset(
        [lfq32trainDataSet, lfq64trainDataSet, tims20211002trainDataSet, Trypsin_HFXtrainDataSet])

    valDataSet, valDataDecoySet = [lfq32valDataSet, lfq64valDataDecoySet, tims20211002valDataSet, Trypsin_HFXvalDataSet], [
        lfq32valDataDecoySet,
        lfq64valDataDecoySet,
        tims20211002valDataSet,
        Trypsin_HFXvalDataDecoySet,
    ]

    trainData, valData, valDataDecoy = GetTrainDataLoader(
        batch_size, trainDataSet, valDataSet, valDataDecoySet
    )
    return trainData, valData, valDataDecoy


def GetTrainDataPath() -> Tuple[str, str, str, str, str, str, str, str]:
    r"""
    获取各训练数据路径, 包含以下各类数据
    * LFQ32fix
    * LFQ64var
    * Trypsin_HFX 血浆
    * tims 血浆
    """
    root = "/data/xp/SpectronautLabelTrainAndTest/"

    lfq32TrainPath = (
        root
        + "LFQ32/trainData/HYE124_TTOF6600_32fix_train_win_data_minmz5_identify.npy"
    )

    lfq32TrainDecoyPath = (
        root
        + "LFQ32/trainData/HYE124_TTOF6600_32fix_train_win_data_minmz5_identify_decoy.npy"
    )

    lfq64TrainPath = (
        root
        + "LFQ64/trainData/HYE124_TTOF6600_64var_train_win_data_minmz5_identify.npy"
    )

    lfq64TrainDecoyPath = (
        root
        + "LFQ64/trainData/HYE124_TTOF6600_64var_train_win_data_minmz5_identify_decoy.npy"
    )

    Trypsin_HFXtrainpath = (
        root
        + "20230111_MN_Plasma_DIA_formal_sample_trypsin_HFX/trainData/20230111_MN_Plasma_DIA_formal_sample_trypsin_HFX_train_win_data_minmz5_identify.npy"
    )

    Trypsin_HFXtrainDecoyPath = (
        root
        + "20230111_MN_Plasma_DIA_formal_sample_trypsin_HFX/trainData/20230111_MN_Plasma_DIA_formal_sample_trypsin_HFX_train_win_data_minmz5_identify_decoy.npy"
    )

    tims20211002TrainPath = (
        root
        + "tims-TOF_20211002_30min_LCH_MN_144-Plasma_SPEED-DIA_300ng_mix-qc/trainData/tims-TOF_20211002_30min_LCH_MN_144-Plasma_SPEED-DIA_300ng_mix-qc_train_win_data_minmz5_identify.npy"
    )

    tims20211002TrainDecoyPath = (
        root
        + "tims-TOF_20211002_30min_LCH_MN_144-Plasma_SPEED-DIA_300ng_mix-qc/trainData/tims-TOF_20211002_30min_LCH_MN_144-Plasma_SPEED-DIA_300ng_mix-qc_train_win_data_minmz5_identify_decoy.npy"
    )

    return (
        lfq32TrainPath,
        lfq32TrainDecoyPath,
        lfq64TrainPath,
        lfq64TrainDecoyPath,
        Trypsin_HFXtrainpath,
        Trypsin_HFXtrainDecoyPath,
        tims20211002TrainPath,
        tims20211002TrainDecoyPath,
    )


def GetTestDataPath() -> (
    Tuple[str, str, str, str, str, str, str, str, str, str, str, str]
):
    r"""
    获取各训练数据路径, 包含以下各类数据
    * LFQ32fix
    * LFQ64var
    * Trypsin_HFX 血浆
    * tims 血浆
    * Hela 数据
    * 无色谱 lfq
    """
    root = "/data/xp/SpectronautLabelTrainAndTest/"

    lfq32TestPath = (
        root + "LFQ32/testData/HYE124_TTOF6600_32fix_test_win_data_minmz5_identify.npy"
    )

    lfq32TestDecoyPath = (
        root
        + "LFQ32/testData/HYE124_TTOF6600_32fix_test_win_data_minmz5_identify_decoy.npy"
    )

    lfq64TestPath = (
        root + "LFQ64/testData/HYE124_TTOF6600_64var_test_win_data_minmz5_identify.npy"
    )

    lfq64TestDecoyPath = (
        root
        + "LFQ64/testData/HYE124_TTOF6600_64var_test_win_data_minmz5_identify_decoy.npy"
    )

    Trypsin_HFXtestpath = (
        root
        + "20230111_MN_Plasma_DIA_formal_sample_trypsin_HFX/testData/20230111_MN_Plasma_DIA_formal_sample_trypsin_HFX_test_win_data_minmz5_identify.npy"
    )

    Trypsin_HFXtestDecoyPath = (
        root
        + "20230111_MN_Plasma_DIA_formal_sample_trypsin_HFX/testData/20230111_MN_Plasma_DIA_formal_sample_trypsin_HFX_test_win_data_minmz5_identify_decoy.npy"
    )

    tims20211002TestPath = (
        root
        + "tims-TOF_20211002_30min_LCH_MN_144-Plasma_SPEED-DIA_300ng_mix-qc/testData/tims-TOF_20211002_30min_LCH_MN_144-Plasma_SPEED-DIA_300ng_mix-qc_test_win_data_minmz5_identify.npy"
    )

    tims20211002TestDecoyPath = (
        root
        + "tims-TOF_20211002_30min_LCH_MN_144-Plasma_SPEED-DIA_300ng_mix-qc/testData/tims-TOF_20211002_30min_LCH_MN_144-Plasma_SPEED-DIA_300ng_mix-qc_test_win_data_minmz5_identify_decoy.npy"
    )

    HelaTestPath = root + "Hela/testData/Hela_test_win_data_minmz5_identify.npy"

    HelaTestDecoyPath = (
        root + "Hela/testData/Hela_test_win_data_minmz5_identify_decoy.npy"
    )

    NoRTtestPath = root + "NoRTLfq/testData/NoRT_LFQ_test_win_data_minmz5_identify.npy"

    NoRTtestDecoyPath = (
        root + "NoRTLfq/testData/NoRT_LFQ_test_win_data_minmz5_identify_decoy.npy"
    )

    return (
        lfq32TestPath,
        lfq32TestDecoyPath,
        lfq64TestPath,
        lfq64TestDecoyPath,
        Trypsin_HFXtestpath,
        Trypsin_HFXtestDecoyPath,
        tims20211002TestPath,
        tims20211002TestDecoyPath,
        HelaTestPath,
        HelaTestDecoyPath,
        NoRTtestPath,
        NoRTtestDecoyPath,
    )


def GetSpectronautResultPath() -> Tuple[str, str, str, str]:
    r"""
    获取 Spectronaut 结果的路径
    """
    root = "/data/xp/SpectronautLabelTrainAndTest/"
    lfq32Path = root + "LFQ32/LFQ32_Spectronaut_HYE124_TTOF6600_32fix_identify_all_pep.npy"
    lfq64Path = root + "LFQ64/LFQ64_Spectronaut_HYE124_TTOF6600_64var_quantify_all_pep.npy"
    tims20211002Path = root + \
        "tims-TOF_20211002_30min_LCH_MN_144-Plasma_SPEED-DIA_300ng_mix-qc/plasma_1_2_3_4_6_7_8_9_Spectronaut_identify_all_pep.npy"
    Trypsin_HFXPath = root + \
        "20230111_MN_Plasma_DIA_formal_sample_trypsin_HFX/20230111_plasma_24QC_Spectronaut_identify_all_pep.npy"
    return lfq32Path, lfq64Path, tims20211002Path, Trypsin_HFXPath
