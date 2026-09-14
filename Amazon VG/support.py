import random
import time

from torch.utils.data import Dataset
import sys
import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from preprocess import serial_asin_category
from extract_img_feature import get_img_feature_pickle


# [代码作用] 将 Amazon 原始 user id 映射成 0...N-1 的连续整数，便于直接索引 user_embedding。
# [注意] 这里只把原始 user ID 映射成连续整数索引，不生成 embedding。
# s_u 来自 user_embedding[user]；user_embedding 在模型初始化时创建，并通过 L_z、L_q 的 BPR 训练不断更新。
def serialize_user(user_set):
    user_set = set(user_set)
    user_idx = 0
    # key: user原始下标，value: user有序下标
    user_serialize_dict = {}
    for user in user_set:
        user_serialize_dict[user] = user_idx
        user_idx += 1
    return user_serialize_dict


# 输入user和item的set，输出user和item从1到n有序的字典
# [代码作用] 同理，把原始 asin/item id 映射成 0...M-1，用来索引 item_embedding（论文 COCE z_v）。
def serialize_item(item_set):
    item_set = set(item_set)
    item_idx = 0
    item_serialize_dict = {}
    for item in item_set:
        item_serialize_dict[item] = item_idx
        item_idx += 1
    return item_serialize_dict


# [论文对应][L_q/L_z] 从“没有与当前 item 交互过的用户”里随机采 1 个负用户 u-。
# [注意] 当前 __getitem__ 实际使用 CSV 中预先保存的 neg_user，这个函数没有在主路径实时调用。
def sample_negative_user(user_set, interaction_user_set):
    users = set(interaction_user_set)
    candidate_users = set(user_set) - set(users)
    return random.sample(list(candidate_users), 1)[0]


# 新建一个user-item的交互字典
# [阅读重点] user -> [item1, item2, ...]。
# 这个方向的字典主要用于构造 L_c 的协同正物品：同一个 user 交互过的其他 item。
def build_user_item_interaction_dict(train_csv='data/train_rating.csv',
                                     user_item_interaction_dict_save='pkl/user_item_interaction_dict.pkl'):
    if os.path.exists(user_item_interaction_dict_save) is True:
        print('从缓存中加载user_item_interaction_dict')
        pkl_file = open(user_item_interaction_dict_save, 'rb')
        data = pickle.load(pkl_file)
        return data['user_item_interaction_dict']
    if os.path.exists("pkl") is False:
        os.makedirs("pkl")
    df = pd.read_csv(train_csv)
    user_item_interaction_dict = {}
    for _, row in tqdm(df.iterrows()):
        movie = row['asin']
        user = row['reviewerID']
        res = user_item_interaction_dict.get(user)
        if res is None:
            user_item_interaction_dict[user] = [movie]
        else:
            res.append(movie)
            user_item_interaction_dict[user] = res
    with open(user_item_interaction_dict_save, 'wb') as file:
        pickle.dump({'user_item_interaction_dict': user_item_interaction_dict}, file)
    return user_item_interaction_dict


# 新建一个item-user的交互字典
# [阅读重点] item -> [user1, user2, ...]。
# 它表达某个物品有哪些历史交互用户，可用于判断/构造 BPR 的正负用户关系。
def build_item_user_interaction_dict(train_csv='data/train_rating.csv',
                                     item_user_interaction_dict_save='pkl/item_user_interaction_dict.pkl'):
    if os.path.exists(item_user_interaction_dict_save) is True:
        print('从缓存中加载', item_user_interaction_dict_save)
        pkl_file = open(item_user_interaction_dict_save, 'rb')
        data = pickle.load(pkl_file)
        return data['item_user_interaction_dict']
    if os.path.exists("pkl") is False:
        os.makedirs("pkl")
    df = pd.read_csv(train_csv)
    item_user_interaction_dict = {}
    for _, row in tqdm(df.iterrows()):
        movie = row['asin']
        user = row['reviewerID']
        res = item_user_interaction_dict.get(movie)
        if res is None:
            item_user_interaction_dict[movie] = [user]
        else:
            res.append(user)
            item_user_interaction_dict[movie] = res
    with open(item_user_interaction_dict_save, 'wb') as file:
        pickle.dump({'item_user_interaction_dict': item_user_interaction_dict}, file)
    return item_user_interaction_dict


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RatingDataset(torch.utils.data.Dataset):
    # ============================================================
    # [阅读重点] 第 3 步重点阅读这个 Dataset，尤其是 __getitem__。
    # 它不仅“读取一条交互”，还提前准备三个训练目标需要的数据：
    #   L_q / L_z：当前正用户 user、负用户 neg_user、当前 item
    #   L_c：positive_item_list、negative_item_list、self_neg_list
    # 最终返回的 8 个变量会原样进入 model.py::train()。
    # ============================================================
    def __init__(self, train_csv, img_features, genres, category_num, user_serialize_dict,  positive_number, negative_number):
        self.train_csv = train_csv
        # 读其他内容
        self.img_feature_dict = img_features
        self.genres_dict = genres
        # print(self.item_pn_df)
        self.user = self.train_csv["reviewerID"]
        self.item = self.train_csv["asin"]
        self.rating = self.train_csv["rating"]

        # [L_q/L_z] CSV 中已预先准备好的负用户。
        self.neg_user = self.train_csv['neg_user']
        self.item_set = set(self.item)

        # 序列化user和item
        self.user_serialize_dict = user_serialize_dict
        self.item_serialize_dict = serialize_item(self.item)

        # 返回个数时，返回全集的user数和训练集的item数
        self.user_number = len(user_serialize_dict)
        self.item_number = len(set(self.item))

        # [L_c] 每个样本的协同正物品数与每组负物品数。
        self.positive_number = positive_number
        self.negative_number = negative_number
        self.category_num = category_num

        # [协同关系索引]
        self.user_item_interaction_dict = build_user_item_interaction_dict()
        self.item_user_interaction_dict = build_item_user_interaction_dict()
        print("整个数据集的user个数为:", self.user_number, "train_set中的用户数目为:", len(set(self.user)))

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, index):
        # ========================================================
        # [一条训练样本的起点]
        # train_csv 的当前行给出一个真实交互 (user, item)，因此 user 是当前正用户 u+。
        # ========================================================
        user = self.user[index]
        item = self.item[index]

        # 处理 item genres
        # [内容输入 X_v：category]
        # 构造长度为 category_num 的向量：当前物品拥有的类别位置赋 1，其余保留 -1。
        # forward() 会利用 != -1 的 mask，只给真实存在的类别分配 attention 权重。
        genres = torch.full((self.category_num, 1), -1)
        genres_index = self.genres_dict.get(item)
        genres = genres.to(device)
        genres[genres_index] = 1
        genres = genres.squeeze(dim=1)

        # 处理 item feature
        # [内容输入 X_v：image] 取当前物品预先提取好的图像特征（后续 model.py 映射为 p_v）。
        img_feature = self.img_feature_dict.get(item)
        get_item_start = time.time()

        # sample neg user spend a lot time.
        interaction_user_set = self.item_user_interaction_dict.get(item)
        # neg_user = sample_negative_user(self.user, interaction_user_set)

        # [论文对应][L_q/L_z] 与当前 item 没有正交互的负用户 u-；这里直接读取预先生成的列。
        neg_user = self.neg_user[index]
        # print('sample neg user:', time.time()-get_item_start)

        # --------------------- #
        #  处理 positive items   #
        #  runtime sampling     #
        # --------------------- #
        # [论文对应][L_c 正物品]
        # 找出当前 user 交互过的物品集合。由于当前样本中的 item 也被该 user 交互过，
        # 从该集合抽到的其他物品与当前 item 至少共享这个 user，因此形成协同共现关系。
        positive_items_ = self.user_item_interaction_dict.get(user)

        # [代码实现细节] 有放回抽 positive_number 个；该集合本身也包含当前 item。
        # 因而这里是官方实现的实际采样方式，阅读时要与论文抽象的 v+ 定义区分开看。
        positive_items = list(np.random.choice(list(positive_items_), self.positive_number, replace=True))
        positive_items_list = [self.item_serialize_dict.get(item) for item in positive_items]

        # runtime sampling negative
        # [论文对应][L_c 负物品]
        # 官方代码从“当前 user 没交互过的 item”中采样负例。
        # 注意：这是代码的具体采样判据；判断实现时以这里为准。
        negative_item_list = []
        neg_item_set = list(self.item_set - set(positive_items_))

        # merge multi negative sample result
        # 一次抽出 positive_number 组负例 + 1 组 self contrast 负例，后面再切分。
        negative_items_ = list(np.random.choice(neg_item_set, self.negative_number*(self.positive_number+1), replace=True))
        negative_items_ = [self.item_serialize_dict.get(it) for it in negative_items_]
        for i in range(self.positive_number):
            start_idx = self.negative_number*i
            end_idx = self.negative_number*(i+1)
            negative_item_list.append(negative_items_[start_idx:end_idx])

        # self neg list 完成 序列化, self的抽样放在和collaborative items中一起抽样负例子，最后分割出来就行了
        # [代码实现细节] 剩余最后 negative_number 个负物品专门供 model.py 的 self contrast 使用。
        self_neg_list = negative_items_[self.positive_number*self.negative_number:]

        # serialize
        # [代码作用] 把原始 user/item id 转为 embedding 表可以直接索引的连续整数 id。
        user = self.user_serialize_dict.get(user)
        item = self.item_serialize_dict.get(item)
        neg_user = self.user_serialize_dict.get(neg_user)

        # [返回值与 model.py::train() 一一对应]
        # user               -> u+，L_q/L_z 正用户
        # item               -> 当前 v
        # genres/img_feature -> X_v 内容侧输入
        # neg_user           -> u-，L_q/L_z 负用户
        # positive_item_list -> L_c 协同正物品
        # negative_item_list -> L_c 每个正物品对应的一组负物品
        # self_neg_list      -> self contrast 负物品
        return torch.tensor(user), torch.tensor(item), genres, torch.tensor(img_feature), torch.tensor(neg_user),\
               torch.tensor(positive_items_list), torch.tensor(negative_item_list), torch.tensor(self_neg_list)


# 测试数据封装
if __name__ == '__main__':
    # [可暂时跳过] 这里是 support.py 自己的简单调试入口，不是正式训练入口。
    print("support.py")

    asin_category_int_map, category_ser_map = serial_asin_category()
    img_feature_dict = get_img_feature_pickle()
    category_length = len(category_ser_map)
    # (self, train_csv, img_features, genres, user_serialize_dict, positive_number, negative_number)
    train_csv = pd.read_csv("data/train_withneg_rating.csv")
    print("ratings.length:", train_csv.__len__())
    all_ratings = pd.read_csv("data/ratings_filter.csv")
    user_ser_dict = serialize_user(all_ratings["reviewerID"])
    dataset = RatingDataset(train_csv, img_feature_dict, asin_category_int_map, category_length, user_ser_dict, 10, 20)
    dataIter = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    it = dataIter.__iter__()
    for i_index in range(10):
        start = time.time()
        u, i, g, i_f, n_user, p_list, n_list, self_n_list = it.next()
        print("time spend:", time.time()-start)
        i_index += 1
    # print(u, i, g, i_f, n_user)
    # print("positive_list, negative_list, self_negative_list")
    # print("genres.shape:", g.shape, "img_f.shape:", i_f.shape)
