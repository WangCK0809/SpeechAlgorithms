"""
@FileName: sampler_generator.py
@Description: Implement addNoise
@Author: Ryuk
@CreateDate: 2020/05/11
@LastEditTime: 2020/05/11
@LastEditors: Please set LastEditors
@Version: v0.1
"""
import numpy as np
from scipy import signal
from numpy.linalg import norm


def add_noise(clean, noise, snr):
    if len(noise) > len(clean):
        noise = noise[:len(clean)]
    else:
        times = len(clean) // len(noise)
        noise = np.tile(noise, times)
        padding = [0] * (len(clean) - len(noise))
        noise = np.hstack([noise, padding])

    noise = noise / norm(noise) * norm(clean) / (10.0 ** (0.05 * snr))
    mix = clean + noise
    return mix


def add_noise_optimize(clean, noise, snr):
    if len(noise) > len(clean):
        noise = noise[:len(clean)]
    else:
        times = len(clean) // len(noise)
        noise = np.tile(noise, times)
        padding = [0] * (len(clean) - len(noise))
        noise = np.hstack([noise, padding])

    # 1> 计算干净语音信号和噪声信号的能量
    clean_power = np.sum(clean ** 2)
    nosie_power = np.sum(noise ** 2)
    # 2> 根据所需信噪比计算目标噪声信号的能量
    target_noise_power = clean_power / (10 ** (snr / 10))
    # 3> 计算噪声信号的缩放因子，并使其具有目标信噪比
    scale_factor = np.sqrt(target_noise_power / nosie_power)
    scaled_noise = noise * scale_factor
    # 4> 生成混合语音信号
    mix = clean + scaled_noise

    return mix


def add_echo(clean, sr, alpha, beta=0.5, delay=0.1, choice=1):
    """
    add echo signal to raw speech
        1. multiply attenuation factor to get attenuated signal
        2. dealy, then add to original signal
    :param clean: clean speech
    :param sr: sample rate
    :param alpha: parameters for type1
    :param beta: parameters for type2
    :param delay: parameters for type2, unit is sencond
    :param choice: echo type
    :return: mix signal
    """
    if choice == 1:
        h = [1]
        h.extend([0] * int(alpha * sr))
        h.extend([0.5])
        h.extend([0] * int(alpha * sr))
        h.extend([0.25])
        mix = signal.convolve(clean, h)
    else:
        mix = clean.copy()
        shift = int(delay * sr)
        for i in range(shift, len(clean)):
            mix[i] = beta * clean[i] + (1 - beta) * clean[i - shift]
    return mix


def add_reverberation(clean, alpha=0.8, R=2000):
    b = [0] * (R + 1)
    b[0], b[-1] = alpha, 1
    a = [0] * (R + 1)
    a[0], a[-1] = 1, 0.5
    mix = signal.filtfilt(b, a, clean)
    return mix


def add_howl(clean, K=0.2):
    g = np.loadtxt("./path.txt")
    c = np.array([0, 0, 0, 0, 1]).T
    h = np.zeros(201)
    h[100] = 1

    xs1 = np.zeros(c.shape[0])
    xs2 = np.zeros(g.shape)
    xs3 = np.zeros(h.shape)

    mix = np.zeros(len(clean))
    temp = 0

    for i in range(len(clean)):
        xs1[1:] = xs1[: - 1]
        xs1[0] = clean[i] + temp
        mix[i] = K * np.dot(xs1.T, c)

        xs3[1:] = xs3[: - 1]
        xs3[0] = mix[i]
        mix[i] = np.dot(xs3.T, h)

        mix[i] = min(1, mix[i])
        mix[i] = max(-1, mix[i])

        xs2[1:] = xs2[: - 1]
        xs2[0] = mix[i]
        temp = np.dot(xs2.T, g)
    return mix
