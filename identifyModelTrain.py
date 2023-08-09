import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import os
from model import *
from DataPretreatment import *
from typing import Tuple, Dict, List
import numpy as np
from tqdm import tqdm


def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def plot_learning_curve(loss_record, title=""):
    total_steps = len(loss_record["train"])
    x = range(total_steps)
    figure, ax1 = plt.subplots()
    ax1.set_title("Learning curve of {}".format(title))
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("TrainLoss")
    ax1.plot(x, loss_record["train"], color="blue", linewidth=1.5)

    # 共用 x 轴
    ax2 = ax1.twinx()
    ax2.set_ylabel("FdrPNum")
    ax2.plot(x, loss_record["fdrPNum"], color="red", linewidth=1.5)

    figure.legend(["TrainLoss", "FdrPNum"], loc="upper right")
    figure.tight_layout()
    figure.savefig("identifyModelLossAcc.png", dpi=1000)


def calculateEvaluateResult(
    Data: DataLoader, model: identifyModel, device
) -> np.ndarray:
    model.eval()
    result = []
    with torch.no_grad():
        for _, (libMsMz, libMz, libSpectrumIfShare, _) in enumerate(Data):
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
    return np.array(result)


def calculateFdrNum(
    testScore: np.ndarray, testDecoyScore: np.ndarray
) -> Tuple[int, int, float, float]:
    TestData = np.sort(testScore)
    decoy = np.sort(testDecoyScore)
    thresh = TestData[0]
    d_i = 0
    for i in tqdm(range(len(TestData))):
        while decoy[d_i] < TestData[i] and d_i < len(decoy) - 1:
            if d_i == len(decoy) - 1:
                break
            d_i += 1
        if i == len(TestData) - 1:
            return 0, 0, 0, 0
        fdr = (len(decoy) - d_i) / (len(TestData) - i)
        if fdr < 0.01:
            thresh = TestData[i]
            break
    if thresh == TestData[0]:
        return 0, 0, 0, 0
    return len(TestData[TestData >= thresh]), len(decoy[decoy >= thresh]), thresh, fdr


def evaluate(
    valData: List[DataLoader],
    valDataDecoy: List[DataLoader],
    model: identifyModel,
    device: torch.device,
) -> Tuple[int, int, float, float]:
    model.eval()
    with torch.no_grad():
        percent = [10, 10, 1, 1]
        pt = [(sum(percent) - p) / sum(percent) for p in percent]
        fdrNumSum = 0
        for i in range(len(valData)):
            testTarget = calculateEvaluateResult(valData[i], model, device)
            decoyTarget = calculateEvaluateResult(
                valDataDecoy[i], model, device)
            fdrPNum, _, _, _ = calculateFdrNum(testTarget, decoyTarget)
            fdrNumSum += pt[i] * fdrPNum
    return fdrNumSum


def train(
    trainSet: DataLoader,
    valSet: List[DataLoader],
    valDecoySet: List[DataLoader],
    model: identifyModel,
    config: Dict[str, any],
    device: torch.device,
    optimizer: torch.optim.Adam,
) -> Tuple[int, Dict[str, List]]:
    model.train()
    maxNum = 0
    n_epochs = config["n_epochs"]
    loss_record = {"train": [], "fdrPNum": []}
    early_stop_cnt = 0
    for epoch in range(n_epochs):
        trainloss = 0
        trainNum = 0
        for libMsMz, libMz, libSpectrumIfShare, libTarget in trainSet:
            optimizer.zero_grad()
            libTarget = libTarget.to(torch.float32)
            libTarget = libTarget.view(-1, 1)
            libMz = torch.unsqueeze(libMz, dim=1)
            libMsMz, libMz, libSpectrumIfShare, libTarget = (
                libMsMz.to(device),
                libMz.to(device),
                libSpectrumIfShare.to(device),
                libTarget.to(device),
            )
            # 前向传递
            y_pred = model(libMz, libMsMz, libSpectrumIfShare)
            # 后向传播
            loss = BCELoss(y_pred, libTarget)
            loss.backward()
            optimizer.step()
            trainloss += loss.item() * libMsMz.size(0)
            trainNum += libTarget.size(0)
            # 验证集测试
        fdrPNum = evaluate(valSet, valDecoySet, model, device)
        if fdrPNum > maxNum:
            maxNum = fdrPNum
            print(
                "Saving model (epoch = {:4d}, trainLoss = {:.4f}, fdrPNum = {:.4f})".format(
                    epoch + 1, trainloss / trainNum, fdrPNum
                )
            )
            torch.save(model.state_dict(), config["save_path"])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        loss_record["train"].append(trainloss / trainNum)
        loss_record["fdrPNum"].append(fdrPNum)
        if early_stop_cnt > config["early_stop"]:
            break
    print("Finish training after {} epochs, fdrPNum = {}".format(epoch + 1, maxNum))
    return maxNum, loss_record


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
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
    # train
    device = get_device()
    net = identifyModel(6, 5, config["attentionDim"])
    net = net.to(device)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)  # 包装为并行风格模型
        print("finish")

    optimizer = optim.Adam(
        params=net.parameters(),
        lr=config["lr"],
        eps=config["eps"],
        betas=config["betas"],
    )
    BCELoss = nn.BCELoss()

    (
        lfq32TrainPath,
        lfq32TrainDecoyPath,
        lfq64TrainPath,
        lfq64TrainDecoyPath,
        Trypsin_HFXtrainpath,
        Trypsin_HFXtrainDecoyPath,
        tims20211002TrainPath,
        tims20211002TrainDecoyPath,
    ) = GetTrainDataPath()

    trainData, valData, valDataDecoy = GetAllTrainData(
        config["batch_size"],
        lfq32TrainPath,
        lfq32TrainDecoyPath,
        lfq64TrainPath,
        lfq64TrainDecoyPath,
        tims20211002TrainPath,
        tims20211002TrainDecoyPath,
        Trypsin_HFXtrainpath,
        Trypsin_HFXtrainDecoyPath,
    )

    min_bce, loss_record = train(
        trainData, valData, valDataDecoy, net, config, device, optimizer
    )
    plot_learning_curve(loss_record, "deep model")
