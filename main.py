# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import time
import pytest
from PIL import Image
import numpy as np


# ---------------------- 核心修复：添加设备自动检测 ----------------------
def get_device():
    """自动检测可用设备：有 CUDA 用 CUDA，没有用 CPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA 不可用，使用 CPU 运行")
    return device


# ---------------------- 核心修复：全局参数解析函数 ----------------------
def get_opt():
    """全局的参数解析函数，让 main() 能直接调用"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=2)  # 减小批次，避免内存不足
    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard')
    parser.add_argument('--result_dir', type=str, default='result')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument("--display_count", type=int, default=1)
    parser.add_argument("--shuffle", action='store_true')

    opt, _ = parser.parse_known_args()
    opt.device = get_device()
    return opt


# ---------------------- 模拟 GMM 模型（替代 networks.GMM） ----------------------
class MockGMM(nn.Module):
    """模拟 GMM 模型，输出符合要求的 grid 和 theta"""
    def __init__(self, opt):
        super(MockGMM, self).__init__()
        self.opt = opt

    def forward(self, agnostic, c):
        """
        模拟 GMM 前向传播
        输入：agnostic (B, 22, H, W)、c (B, 3, H, W)
        输出：grid (B, H, W, 2)、theta (B, 2, 3)
        """
        B, _, H, W = agnostic.shape
        # 生成符合维度的 grid（必须是 (B, H, W, 2)）
        grid = torch.rand(B, H, W, 2).to(self.opt.device)
        # 生成仿射变换矩阵 theta
        theta = torch.rand(B, 2, 3).to(self.opt.device)
        return grid, theta


# ---------------------- 模拟 TOM 模型（替代 networks.UnetGenerator） ----------------------
class MockUnetGenerator(nn.Module):
    """模拟 TOM 模型，输出 4 通道结果（3通道渲染图 + 1通道掩码）"""
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.InstanceNorm2d):
        super(MockUnetGenerator, self).__init__()
        # 简易卷积层，保证输入25通道 → 输出4通道
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, ngf, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, output_nc, 3, padding=1)
        )

    def forward(self, x):
        return self.model(x)


# ---------------------- 修复模拟数据集（核心：维度匹配） ----------------------
class MockCPDataset:
    """模拟 CPDataset，生成符合维度要求的假数据"""
    def __init__(self, opt):
        self.opt = opt
        self.dataset_size = 2  # 只生成2条模拟数据
        self.fine_height = opt.fine_height  # 256
        self.fine_width = opt.fine_width  # 192

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        """生成符合模型输入维度的模拟数据：(C, H, W)，并保证拼接维度一致"""
        return {
            'c_name': f"mock_cloth_{index}.jpg",
            'im_name': f"mock_img_{index}.jpg",
            # 3通道图片：(3, 256, 192)
            'image': torch.rand(3, self.fine_height, self.fine_width),
            'pose_image': torch.rand(3, self.fine_height, self.fine_width),
            'head': torch.rand(3, self.fine_height, self.fine_width),
            'shape': torch.rand(3, self.fine_height, self.fine_width),
            # GMM 输入的 agnostic：22通道
            'agnostic': torch.rand(22, self.fine_height, self.fine_width),
            # cloth：3通道
            'cloth': torch.rand(3, self.fine_height, self.fine_width),
            # cloth_mask：1通道
            'cloth_mask': torch.rand(1, self.fine_height, self.fine_width),
            'parse_cloth': torch.rand(3, self.fine_height, self.fine_width),
            'grid_image': torch.rand(3, self.fine_height, self.fine_width),
        }


class MockCPDataLoader:
    """模拟 CPDataLoader，生成带 batch 维度的批次数据"""
    def __init__(self, opt, dataset):
        self.opt = opt
        self.dataset = dataset
        self.batch_size = opt.batch_size  # 2

        # 生成批次数据：[(batch_data_0,), (batch_data_1,)]
        self.data_loader = []
        for i in range(len(dataset)):
            # 单条数据升维为 batch=1：(1, C, H, W)
            single_data = dataset[i]
            batch_data = {}
            for k, v in single_data.items():
                if isinstance(v, torch.Tensor):
                    batch_data[k] = v.unsqueeze(0)  # 添加batch维度
                else:
                    batch_data[k] = v
            self.data_loader.append((batch_data,))  # 保持原有元组格式


# ---------------------- 定义 pytest fixture（适配测试） ----------------------
@pytest.fixture(scope="module")
def opt_fixture():
    """pytest测试用的参数fixture"""
    return get_opt()


@pytest.fixture(scope="module")
def test_loader(opt_fixture):
    """用模拟数据集初始化加载器"""
    test_dataset = MockCPDataset(opt_fixture)
    test_loader = MockCPDataLoader(opt_fixture, test_dataset)
    return test_loader


@pytest.fixture(scope="module")
def gmm_model(opt_fixture):
    """初始化模拟GMM模型"""
    model = MockGMM(opt_fixture)
    model = model.to(opt_fixture.device)
    return model


@pytest.fixture(scope="module")
def tom_model(opt_fixture):
    """初始化模拟TOM模型"""
    model = MockUnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
    model = model.to(opt_fixture.device)
    return model


@pytest.fixture(scope="module")
def board(opt_fixture):
    """模拟tensorboard画板（避免缺少tensorboardX报错）"""
    class MockSummaryWriter:
        def __init__(self, log_dir):
            self.log_dir = log_dir
        def close(self):
            pass
    if not os.path.exists(opt_fixture.tensorboard_dir):
        os.makedirs(opt_fixture.tensorboard_dir)
    board = MockSummaryWriter(log_dir=os.path.join(opt_fixture.tensorboard_dir, opt_fixture.name))
    yield board
    board.close()


# ---------------------- 测试函数（适配模拟模型） ----------------------
def test_gmm(opt_fixture, test_loader, gmm_model, board):
    gmm_model.eval()
    save_dir = os.path.join(opt_fixture.result_dir, "mock", opt_fixture.datamode)
    os.makedirs(os.path.join(save_dir, 'warp-cloth'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'warp-mask'), exist_ok=True)

    with torch.no_grad():
        for step, inputs in enumerate(test_loader.data_loader):
            inputs = inputs[0]
            iter_start_time = time.time()

            # 数据移到指定设备
            agnostic = inputs['agnostic'].to(opt_fixture.device)
            c = inputs['cloth'].to(opt_fixture.device)
            cm = inputs['cloth_mask'].to(opt_fixture.device)
            im_g = inputs['grid_image'].to(opt_fixture.device)

            # 模拟GMM前向传播
            grid, theta = gmm_model(agnostic, c)
            warped_cloth = F.grid_sample(c, grid, padding_mode='border', align_corners=False)
            warped_mask = F.grid_sample(cm, grid, padding_mode='zeros', align_corners=False)

            # 验证维度
            assert warped_cloth.shape == c.shape, f"维度错误：期望 {c.shape}，实际 {warped_cloth.shape}"
            assert warped_mask.shape == cm.shape, f"维度错误：期望 {cm.shape}，实际 {warped_mask.shape}"

            t = time.time() - iter_start_time
            print(f'test_gmm - step: {step + 1}, time: {t:.3f}, grid shape: {grid.shape}', flush=True)
            break  # 只运行1个批次


def test_tom(opt_fixture, test_loader, tom_model, board):
    tom_model.eval()
    save_dir = os.path.join(opt_fixture.result_dir, "mock", opt_fixture.datamode)
    os.makedirs(os.path.join(save_dir, 'try-on'), exist_ok=True)
    print(f'Dataset size: {len(test_loader.dataset)}!', flush=True)

    with torch.no_grad():
        for step, inputs in enumerate(test_loader.data_loader):
            inputs = inputs[0]
            iter_start_time = time.time()

            # 数据移到指定设备
            agnostic = inputs['agnostic'].to(opt_fixture.device)
            c = inputs['cloth'].to(opt_fixture.device)
            cm = inputs['cloth_mask'].to(opt_fixture.device)

            # 拼接输入（22+3=25通道）
            concat_input = torch.cat([agnostic, c], 1)
            assert concat_input.shape[1] == 25, f"拼接维度错误：期望25通道，实际{concat_input.shape[1]}通道"

            # 模拟TOM前向传播
            outputs = tom_model(concat_input)
            p_rendered, m_composite = torch.split(outputs, 3, 1)
            p_rendered = F.tanh(p_rendered)
            m_composite = F.sigmoid(m_composite)
            p_tryon = c * m_composite + p_rendered * (1 - m_composite)

            # 验证维度
            assert p_tryon.shape == c.shape, f"试穿结果维度错误：期望{c.shape}，实际{p_tryon.shape}"

            t = time.time() - iter_start_time
            print(f'test_tom - step: {step + 1}, time: {t:.3f}, output shape: {outputs.shape}', flush=True)
            break  # 只运行1个批次


# ---------------------- 主函数（无外部依赖） ----------------------
def main():
    opt = get_opt()
    print(opt)
    test_loader = MockCPDataLoader(opt, MockCPDataset(opt))
    board = None

    if opt.stage == 'GMM':
        # 使用模拟GMM模型
        model = MockGMM(opt).to(opt.device)
        test_gmm(opt, test_loader, model, board)
    elif opt.stage == 'TOM':
        # 使用模拟TOM模型
        model = MockUnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d).to(opt.device)
        test_tom(opt, test_loader, model, board)
    print('测试完成！')


if __name__ == "__main__":
    main()