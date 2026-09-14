import numpy as np


def get_img_feature_pickle(img_feature_path='data/img_feature.npy'):
    # ============================================================
    # [代码作用] 读取已经预先提取好的物品图像特征。
    # key   = item 的 asin
    # value = 该 item 的图像特征向量
    #
    # [论文对应] 这是 X_v 中 image 侧信息的输入来源。
    # 这里本身不训练图像网络；model.py 再通过可学习的 image_projection
    # 将原始图像特征映射到推荐模型使用的 embedding 空间。
    # ============================================================
    # key: item的asin, value是item的图像特征
    return np.load(img_feature_path, allow_pickle=True).item()
