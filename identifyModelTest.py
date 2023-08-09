import torch.nn as nn
import matplotlib.pyplot as plt
from model import *
from DataPretreatment import *
from typing import Tuple, Dict, List
import numpy as np
from tqdm import tqdm


def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def CalculateTheOverlap(SpectronautPath: str, DeepLearningResult: np.ndarray, thresh: float, underScoreIf: bool = False) -> int:
    r"""
    Input:
    * SpectronautPath: Spectronaut 结果的路径
    * DeepLearningResult: 深度学习模型打分情况

    1. 统计每个文件下 Spectronaut 和 DeepLearning 定性结果
    2. 对每个文件计算 Overlap
    3. 返回每个文件 Overlap 的平均值

    Output: return
    * 每个文件 Overlap 的平均值
    """
    # 加载的文件中包含单个对象或字典时才有效, 将其转为该对象或字典。
    SpectronautResult: Dict[str, Dict[Tuple[str, int], List]] = np.load(
        SpectronautPath, allow_pickle=True).item()
    # 获取检测的文件名称
    filePath = list(SpectronautResult.keys())
    # 和 Spectronaut 的存储分布一致
    fileDeepLearningResult = {file: {} for file in filePath}
    # deeplearning 打分情况
    libScoreList = DeepLearningResult[0]
    # 肽段情况 (fileName, window, (libName, Charge))
    libFileWinList = DeepLearningResult[1]
    for index, libScore in enumerate(libScoreList):
        if libScore >= thresh:
            fileName = libFileWinList[index][0]
            libNameCharge = libFileWinList[index][2]
            if underScoreIf:
                libName = '_' + libNameCharge[0] + '_'
                charge = libNameCharge[1]
                libNameCharge = (libName, charge)
            fileDeepLearningResult[fileName][libNameCharge] = libScore
    # 计算每个文件中 DeepLearning 和 Spectronaut 结果的 Overlap 大小
    Overlap = {}
    for file in SpectronautResult.keys():
        Overlap[file] = len(set(SpectronautResult[file]) &
                            set(fileDeepLearningResult[file]))
    # 计算 Spectronaut 在每个文件下检测到的肽段数量
    SpectronautLibNum = {file: 0 for file in SpectronautResult.keys()}
    for file in SpectronautResult.keys():
        SpectronautLibNum[file] = len(SpectronautResult[file])
    # 计算 DeepLearning 在每个文件下检测到的肽段数量
    DeepLearningLibNum = {file: 0 for file in SpectronautResult.keys()}
    for file in SpectronautResult.keys():
        DeepLearningLibNum[file] = len(fileDeepLearningResult[file])
    # 统计两种方法检测的平均值和 Overlap
    print("DeepLearningAverage: {}, SpectronautAverage: {}, Overlap: {}".format(sum(list(DeepLearningLibNum.values())) / len(fileDeepLearningResult.keys()), sum(
        list(SpectronautLibNum.values())) / len(SpectronautResult.keys()), sum(list(Overlap.values())) / len(Overlap)))


def GenerateFdrResultFig(test: np.ndarray, decoy: np.ndarray, savePath: str):
    r"""
    Input:
    * test 测试集的结果
    * decoy 测试诱饵集的结果
    * savePath 打分情况图保存路径

    生成模型在目标库和诱饵库上的打分柱状图, 横坐标为 0 ~ 1 的分数, 纵坐标为处于这个分数段的肽段数

    橙色柱状图为诱饵库打分情况, 粉色柱状图为目标库打分情况, 棕色为它们相交部分
    """
    testTarget, _ = test[0], test[1]
    decoyTarget, _ = decoy[0], decoy[1]

    test = np.sort(testTarget)
    decoy = np.sort(decoyTarget)
    thresh = test[0]
    d_i = 0

    for i in tqdm(range(len(test))):
        while decoy[d_i] < test[i]:
            d_i += 1
        fdr = (len(decoy) - d_i) / (len(test) - i)
        if fdr < 0.01:
            thresh = test[i]
            break
    print(thresh, fdr, i, d_i, len(
        decoy[decoy >= thresh]), len(test[test >= thresh]))

    _, ax = plt.subplots()
    testHist, testBins = np.histogram(test, bins=np.arange(0, 1.02, 0.02))
    decoyHist, decoyBins = np.histogram(decoy, bins=np.arange(0, 1.02, 0.02))
    OverLap = [min(testHist[i], decoyHist[i])
               for i in range(len(testBins[:-1]))]
    index = np.searchsorted(np.arange(0, 1.02, 0.02), thresh)
    ax.vlines(
        x=np.arange(0, 1.02, 0.02)[index],
        ymin=0,
        ymax=max(np.max(testHist), np.max(decoyHist)),
        linestyles=":",
        color="#D85E03",
        label="$fdr=%.2f$" % thresh,
    )
    width = 0.02
    ax.bar(testBins[:-1] + width / 2, testHist, width=width, color="salmon")
    ax.bar(decoyBins[:-1] + width / 2, decoyHist, width=width, color="orange")
    ax.bar(testBins[:-1] + width / 2, OverLap, width=width, color="chocolate")
    plt.legend(loc="best")
    plt.title("target and decoy score figure")
    plt.savefig(savePath + ".png", dpi=1000)
    return thresh


def test(
    testData: DataLoader,
    fileWinLib: np.ndarray,
    device: torch.device,
    model: identifyModel,
    savePath: str,
) -> np.ndarray:
    model.eval()
    result = []
    with torch.no_grad():
        for _, (libMsMz, libMz, libSpectrumIfShare, _) in enumerate(testData):
            libMz = torch.unsqueeze(libMz, dim=1)
            libMsMz, libMz, libSpectrumIfShare = (
                libMsMz.to(device),
                libMz.to(device),
                libSpectrumIfShare.to(device),
            )
            y_pred = model(libMz, libMsMz, libSpectrumIfShare)
            y_pred = y_pred.view(-1)
            y_pred = y_pred.tolist()
            result += y_pred
    TestResult = [np.array(result), fileWinLib]
    np.save(savePath + ".npy", TestResult)
    return TestResult


if __name__ == "__main__":
    config = {
        "lr": 0.00007,
        "betas": (0.5, 0.9),
        "eps": 1e-7,
        "attentionDim": 128,
        "batch_size": 2048,
        "n_epochs": 1000,
        "early_stop": 200,
        "save_path": "models/model.pth",
    }

    # test
    (
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
    ) = GetTestDataPath()
    # the result path of Spectronaut
    (
        lfq32Spectronaut,
        lfq64Spectronaut,
        tims20211002Spectronaut,
        Trypsin_HFXspectronaut
    ) = GetSpectronautResultPath()

    lfq32Data, lfq32DataDecoy, lfq32FileWinLib, lfq32FileWinLibDecoy = GetTestData(
        config["batch_size"], lfq32TestPath, lfq32TestDecoyPath
    )

    lfq64Data, lfq64DataDecoy, lfq64FileWinLib, lfq64FileWinLibDecoy = GetTestData(
        config["batch_size"], lfq64TestPath, lfq64TestDecoyPath
    )

    (
        Trypsin_HFXdata,
        Trypsin_HFXdataDecoy,
        Trypsin_HFXFileWinLib,
        Trypsin_HFXFileWinLibDecoy,
    ) = GetTestData(config["batch_size"], Trypsin_HFXtestpath, Trypsin_HFXtestDecoyPath)

    (
        tims20211002Data,
        tims20211002DataDecoy,
        tims20211002FileWinLib,
        tims20211002FileWinLibDecoy,
    ) = GetTestData(
        config["batch_size"], tims20211002TestPath, tims20211002TestDecoyPath
    )

    HelaData, HelaDataDecoy, HelaFileWinLib, HelaFileWinLibDecoy = GetTestData(
        config["batch_size"], HelaTestPath, HelaTestDecoyPath
    )

    NoRTdata, NoRTdataDecoy, NoRTFileWinLib, NoRTFileWinLibDecoy = GetTestData(
        config["batch_size"], NoRTtestPath, NoRTtestDecoyPath
    )

    device = get_device()
    net = identifyModel(6, 5, config["attentionDim"])
    net = net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)  # 包装为并行风格模型
        print("finish")
    # 加载训练后保存的最优模型
    BestModel = torch.load(config["save_path"])
    net.load_state_dict(BestModel)

    lfq32Test = test(lfq32Data, lfq32FileWinLib, device, net, "lfq32Result")
    lfq32Decoy = test(
        lfq32DataDecoy, lfq32FileWinLibDecoy, device, net, "lfq32DecoyResult"
    )
    print("LFQ32Result")
    thresh = GenerateFdrResultFig(lfq32Test, lfq32Decoy, "lfq32")
    CalculateTheOverlap(lfq32Spectronaut, lfq32Test, thresh)
    plt.cla()

    lfq64Test = test(lfq64Data, lfq64FileWinLib, device, net, "lfq64Result")
    lfq64Decoy = test(
        lfq64DataDecoy, lfq64FileWinLibDecoy, device, net, "lfq64DecoyResult"
    )
    print("LFQ64Result")
    thresh = GenerateFdrResultFig(lfq64Test, lfq64Decoy, "lfq64")
    CalculateTheOverlap(lfq64Spectronaut, lfq64Test, thresh)
    plt.cla()

    Trypsin_HFXtest = test(
        Trypsin_HFXdata, Trypsin_HFXFileWinLib, device, net, "Trypsin_HFXresult"
    )
    Trypsin_HFXdecoy = test(
        Trypsin_HFXdataDecoy,
        Trypsin_HFXFileWinLibDecoy,
        device,
        net,
        "Trypsin_HFXdecoyResult",
    )
    print("TrypsinResult")
    thresh = GenerateFdrResultFig(
        Trypsin_HFXtest, Trypsin_HFXdecoy, "Trypsin_HFX")
    CalculateTheOverlap(Trypsin_HFXspectronaut, Trypsin_HFXtest, thresh, True)
    plt.cla()

    tims20211002Test = test(
        tims20211002Data, tims20211002FileWinLib, device, net, "tims20211002Result"
    )
    tims20211002Decoy = test(
        tims20211002DataDecoy,
        tims20211002FileWinLibDecoy,
        device,
        net,
        "tims20211002DecoyResult",
    )
    print("tims20211002Result")
    thresh = GenerateFdrResultFig(
        tims20211002Test, tims20211002Decoy, "tims20211002")
    CalculateTheOverlap(tims20211002Spectronaut,
                        tims20211002Test, thresh, True)
    plt.cla()

    HelaTest = test(HelaData, HelaFileWinLib, device, net, "HelaResult")
    HelaDecoy = test(HelaDataDecoy, HelaFileWinLibDecoy,
                     device, net, "HelaDecoyResult")
    GenerateFdrResultFig(HelaTest, HelaDecoy, "Hela")
    plt.cla()

    NoRTtest = test(NoRTdata, NoRTFileWinLib, device, net, "NoRTresult")
    NoRTdecoy = test(NoRTdataDecoy, NoRTFileWinLibDecoy,
                     device, net, "NoRTdecoyResult")
    GenerateFdrResultFig(NoRTtest, NoRTdecoy, "NoRT")
    plt.cla()
