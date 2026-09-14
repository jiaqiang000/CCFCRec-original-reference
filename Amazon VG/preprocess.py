import pickle
import os
import pandas as pd


def serial_asin_category(pkl_name='data/asin_int_category.pkl'):
    # ============================================================
    # [代码作用] 把 Amazon 原始字符串 category 转换成连续整数 id。
    # 最终得到：
    #   asin_category_int_map：item asin -> [category_id1, category_id2, ...]
    #   category_ser_map：category 字符串 -> category_id
    # [论文对应] 这是 X_v 中 category/attribute 侧信息进入模型前的预处理，
    # 真正的可学习 category embedding 在 model.py 的 attr_matrix 中。
    # ============================================================
    if os.path.exists(pkl_name) is True:
        # [缓存] 若已经处理过就直接读取，避免每次训练重复构造映射。
        pkl_file = open(pkl_name, "rb")
        data = pickle.load(pkl_file)
        # data['asin_category_int_map']， asin: category, category为经过顺序化后的属性
        # data['category_ser_map']， category: category_int_num, category对应的顺序编号
        return data['asin_category_int_map'], data['category_ser_map']

    # [预处理步骤 1] 收集数据集中出现过的全部 category。
    asin_df = pd.read_csv("data/asin.csv")
    category_set = set([])
    for idx, row in asin_df.iterrows():
        cat = row['category'].split(',')
        for i in cat:
            category_set.add(i)

    # 用顺序给category编上序号，把asin中的category字符串转换为数字
    # [预处理步骤 2] category 字符串 -> 连续整数 id。
    idx = 0
    category_ser_map = {}
    for it in category_set:
        category_ser_map[it] = idx
        idx += 1

    # [预处理步骤 3] 为每个 item/asin 保存它拥有的 category id 列表。
    asin_category_int_map = {}
    for idx, row in asin_df.iterrows():
        cat = row['category'].split(',')
        asin = row['asin']
        tmp_list = []
        for i in cat:
            tmp_list.append(category_ser_map.get(i))
        asin_category_int_map[asin] = tmp_list

    # [缓存] 下次直接加载。
    with open(pkl_name, "wb") as file:
        pickle.dump({'asin_category_int_map': asin_category_int_map, 'category_ser_map': category_ser_map}, file)
