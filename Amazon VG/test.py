import random

import pandas as pd
import torch
import numpy as np
import time
import os
from myargs import get_args
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# [基线工具] 随机推荐 k 个用户；不属于 CCFCRec 的核心推理路径。
def get_random_user_rank_list(model, genres, image_feature, k):
    user_number = model.user_embedding.shape[0]
    user_list = list(range(0, user_number))
    res_list = []
    for i in range(k):
        res_list.append(random.sample(user_list, 1)[0])
    return res_list


def get_similar_user_speed(model, genres, image_feature, k):
    # ============================================================
    # [阅读重点] 严格冷启动测试最关键的函数。
    # 对测试冷物品，没有可用历史交互，因此这里【不使用 item_embedding】来表示测试物品。
    # Figure 2 在测试时留下的主路径就是：
    #   X_v(category + image) -> q_v(CBCE) -> 与 UCE s_u 做内积 -> 排序
    # ============================================================
    genres = genres.unsqueeze(dim=0)
    img_feature = image_feature.unsqueeze(dim=0)

    # [论文符号] 只根据冷物品可观察内容生成 q_v（代码 q_v_c）。
    q_v_c = model(genres, img_feature, 1)

    # [论文符号] 所有用户的 UCE s_u。
    # 用户是训练期已有用户，因此 user_embedding 可以在测试时继续使用。
    user_emb = model.user_embedding

    # [论文对应] f_q(s_u, q_v) 使用内积作为预测分数。
    # 对当前冷物品，一次性计算它与全部用户的得分。
    ratings = torch.mul(user_emb, q_v_c).sum(dim=1)

    # 分数从高到低排序，取 Top-K 用户。
    index = torch.argsort(-ratings)
    return index[0:k].cpu().detach().numpy().tolist()


# [评估指标] 当前 item 的真实交互用户与 Top-K 推荐用户的命中数。
def hr_at_k(item, recommend_users, item_user_dict, k):
    groundtruth_user = item_user_dict.get(item)
    recommend_users = recommend_users[0:k]
    inter = set(groundtruth_user).intersection(set(recommend_users))
    return len(inter)


# [评估指标工具] Discounted Cumulative Gain。
def dcg_k(r):
    r = np.asarray(r)
    val = np.sum((np.power(2, r) - 1) / (np.log2(np.arange(1+1, r.size + 2))))
    return val


# [评估指标] NDCG 不只关心是否命中，也考虑命中的用户是否排在推荐列表更前面。
def ndcg_k(item, recommend_users, item_user_dict, k):
    groundtruth_user = item_user_dict.get(item)
    recommend_users = recommend_users[0:k]
    ratings = []
    ndcg = 0.0
    for u in recommend_users:
        if u in groundtruth_user:
            ratings.append(1.0)
        else:
            ratings.append(0.0)
    ratings_ideal = sorted(ratings, reverse=True)
    ideal_dcg = dcg_k(ratings_ideal)
    if ideal_dcg != 0:
        ndcg = (dcg_k(ratings) / ideal_dcg)
    return ndcg


class Validate:
    def __init__(self, validate_csv, user_serialize_dict, img, genres, category_num):
        print("validate class init")
        validate_csv = pd.read_csv(validate_csv)

        # [严格冷启动评估对象] validation/test 中的 item 集合。
        self.item = set(validate_csv['asin'])
        self.item_user_dict = {}

        # 构建完成 item->user dict
        # [Ground Truth] 保存每个测试 item 实际被哪些用户交互，用于之后计算 HR/NDCG；
        # 注意这个 ground truth 只用于“评估答案”，并没有输入模型来生成测试 item 表示。
        for it in self.item:
            users = validate_csv[validate_csv['asin'] == it]['reviewerID']
            users = [user_serialize_dict.get(u) for u in users]
            self.item_user_dict[it] = users
        self.img_dict = img
        self.genres_dict = genres
        self.category_num = category_num

    def start_validate(self, model):
        # ============================================================
        # [阅读重点] 对每一个验证冷物品：
        #   1) 只取 category + image
        #   2) 内容生成 CBCE q_v
        #   3) q_v 与所有 user_embedding 内积
        #   4) 排出 Top-20 用户
        #   5) 用隐藏的真实交互计算 HR/NDCG
        # ============================================================
        # 开始评估
        hr_hit_cnt_5, hr_hit_cnt_10, hr_hit_cnt_20 = 0, 0, 0
        ndcg_sum_5, ndcg_sum_10, ndcg_sum_20 = 0.0, 0.0, 0.0
        max_k = 20
        it_idx = 0
        for it in self.item:
            # 输出
            model = model.to(device)  # move to cpu

            # 处理 item genres
            # [测试时可观察侧信息] 当前冷物品的 category。
            genres = torch.full((self.category_num, 1), -1)
            genres_index = self.genres_dict.get(it)
            genres[genres_index] = 1
            genres = genres.squeeze(dim=1)
            genres = torch.tensor(genres)

            # [测试时可观察侧信息] 当前冷物品的 image feature。
            image_feature = self.img_dict.get(it)
            image_feature = torch.tensor(image_feature)
            genres = genres.to(device)
            image_feature = image_feature.to(device)

            # [严格冷启动核心] 这里生成推荐时只把 genres/image_feature 交给模型，
            # 没有把测试 item 的历史交互或 item_embedding 传进去。
            with torch.no_grad():
                recommend_users = get_similar_user_speed(model, genres, image_feature, max_k)

            # 计算hr指标
            hr_hit_cnt_5 += hr_at_k(it, recommend_users, self.item_user_dict, 5)
            hr_hit_cnt_10 += hr_at_k(it, recommend_users, self.item_user_dict, 10)
            hr_hit_cnt_20 += hr_at_k(it, recommend_users, self.item_user_dict, 20)

            # 计算NDCG指标
            ndcg_sum_5 += ndcg_k(it, recommend_users, self.item_user_dict, 5)
            ndcg_sum_10 += ndcg_k(it, recommend_users, self.item_user_dict, 10)
            ndcg_sum_20 += ndcg_k(it, recommend_users, self.item_user_dict, 20)
            # print("评估进度:", it_idx, "/", len(item))
            it_idx += 1

        item_len = len(self.item)
        hr_5 = hr_hit_cnt_5 / (item_len * 5)
        hr_10 = hr_hit_cnt_10 / (item_len * 10)
        hr_20 = hr_hit_cnt_20 / (item_len * 20)
        ndcg_5 = ndcg_sum_5/item_len
        ndcg_10 = ndcg_sum_10/item_len
        ndcg_20 = ndcg_sum_20/item_len
        print("hr@5:", "hr_10:", "hr_20:", 'ndcg@5', 'ndcg@10', 'ndcg@20')
        print(hr_5, ',', hr_10, ',', hr_20, ',', ndcg_5, ',', ndcg_10, ',', ndcg_20)
        return hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20


if __name__ == '__main__':
    # ============================================================
    # [可暂时跳过] 独立运行 test.py 的模型加载/测试入口。
    # 理解“严格冷启动为什么只依赖内容”时，上面的 get_similar_user_speed + start_validate 才是重点。
    # ============================================================
    # 参数解析器
    # 参数解析器
    import pickle
    from support import RatingDataset
    from model import CCFCRec
    args = get_args()
    # 提取user的原id: 序列化id的dict
    train_path = "data/train_withneg_rating.csv"
    vliad_path = 'data/test_rating.csv'
    train_df = pd.read_csv(train_path)
    load_dir = 'result/2022-10-14/'
    pkl_file = open(load_dir+'save_dict.pkl', 'rb')
    data = pickle.load(pkl_file)
    dataSet = RatingDataset(train_df, data['img_feature_dict'], data['asin_category_int_map'], data['category_ser_map_len'],
                            data['user_ser_dict'], args.positive_number, args.negative_number)
    args.user_number = dataSet.user_number
    args.item_number = dataSet.item_number
    validator = Validate(validate_csv=vliad_path, user_serialize_dict=data['user_ser_dict'], img=data['img_feature_dict'],
                         genres=data['asin_category_int_map'], category_num=data['category_ser_map_len'])
    myModel = CCFCRec(args)
    print('---------数据集加载完毕，开始测试----------------')
    test_result_name = 'test_result.csv'
    with open(test_result_name, 'a+') as f:
        f.write("p@5,p@10,p@20,ndcg@5,ndcg@10,ndcg@20\n")
    load_array = ['98', '99', '100']
    for model in load_array:
        myModel.load_state_dict(torch.load(load_dir+'/'+model+'.pt'))
        hr5, hr_10, hr_20, n_5, n_10, n_20 = validator.start_validate(myModel)

        # [注意] 官方原代码此处 format 使用 p5 / p_10 / p_20，
        # 但本段上面实际接收的变量名是 hr5 / hr_10 / hr_20，疑似原实现变量名遗留问题。
        # 为保持 reference 代码逻辑不变，这里只标注，不修复。
        with open(test_result_name, 'a+') as f:
            f.write("{},{},{},{},{},{}\n".format(p5, p_10, p_20, n_5, n_10, n_20))
