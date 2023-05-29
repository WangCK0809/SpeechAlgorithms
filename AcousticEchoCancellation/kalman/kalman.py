"""
@FileName: kalman.py
@Description: Implement kalman
@Author: Ryuk
@CreateDate: 2021/08/26
@LastEditTime: 2021/09/05
@LastEditors: Please set LastEditors
@Version: v0.1
"""
import numpy as np
import librosa
from tqdm import tqdm
import soundfile as sf

far, sr = librosa.load("../samples/far.wav", sr=16000)  # 加载远端音频信号
near, sr = librosa.load("../samples/near.wav", sr=16000)    # 加载近端音频信号

L = 256     # 系统阶数
P = 1       # 单位阵阶数
delta = 0.0001      # Rmu 的参数
w_cov = 0.01        # 协方差矩阵的参数
v_conv = 0.1        # 协方差矩阵的参数
sigma_e = 0.001     # 估计的残差方差
sigma_x = 0.001     # 输入信号的方差
alpha = 0.9         # 协方差矩阵的更新速率
lambda_v = 0.999    # 更新协方差矩阵参数的速率

h = np.zeros((L, 1))    # 初始化滤波器系数
h_hat = np.zeros((L, 1))    # 估计的滤波器系数

IL = np.identity(L)     # 创建 L × L 的单位矩阵
IP = np.identity(P)     # 创建 P × P 的单位矩阵

Rm = np.zeros((L, L))   # 初始化协方差矩阵 Rm
Rmu = delta * IL        # 初始化 Rmu
Rex = 1e-3 * np.ones((L, 1))    # 初始化 Rex

frame_num = len(far) // L   # 迭代的帧数

e = np.zeros(len(far))      # 存储误差信号
for i in tqdm(range(len(far) - L)):     # 迭代处理每个帧的数据
    X = np.expand_dims(far[i:i+L], axis=1)  # 扩展维度，将输入信号 X 转换为 L × 1 的数组
    Rm = Rmu + w_cov * IL   # 更新协方差矩阵 Rm
    Re = X.T @ Rm @ X + v_conv * IP     # 计算估计误差的方差
    K = Rm @ X / (Re + 0.03)            # 计算增益矩阵 K
    e[i] = near[i+L] - X.T @ h_hat      # 计算误差信号 e
    h_old = h_hat                       # 保存上一次的滤波器系数估计
    h_hat = h_hat + K * e[i]            # 更新滤波器系数估计
    Rmu = (IL - K @ X.T) * Rm           # 更新 Rmu
    delat_h = h_hat - h_old             # 计算滤波器系数估计的变化量
    w_cov = alpha * w_cov + (1 - alpha) * (delat_h.T @ delat_h)     # 更新协方差矩阵参数 w_cov
    Rex = lambda_v * Rex + (1 - lambda_v) * X * e[i]        # 更新 Rex
    sigma_x = lambda_v * sigma_x + (1 - lambda_v) * X[-1] * X[-1]   # 更新输入信号方差
    sigma_e = lambda_v * sigma_e + (1 - lambda_v) * e[i] * e[i]     # 更新估计的残差方差
    v_conv = sigma_e - (1/(sigma_x + 0.03) * (Rex.T @ Rex))         # 更新协方差矩阵参数 v_conv

sf.write("./kalman_out.wav", e, sr)     # 将误差信号保存为音频文件





