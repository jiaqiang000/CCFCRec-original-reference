import math
import os
import sys
import pickle
import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from preprocess import serial_asin_category
from extract_img_feature import get_img_feature_pickle
from support import RatingDataset
from tqdm import tqdm
import pandas as pd
import time
from support import serialize_user
from test import Validate
from myargs import get_args, args_tostring


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#torch.cuda.set_device(1)


# CCFCRec
class CCFCRec(nn.Module):
    def __init__(self, args):
        super(CCFCRec, self).__init__()

        # ============================================================
        # [阅读重点] 第 1 步：先看 __init__，搞清楚模型里有哪些可学习参数。
        # 对照论文 Figure 2，可以先把参数分成三部分：
        #   1) 内容侧：attribute/image 参数 + gen_layer1/2，对应 X_v -> c_v -> g_c -> q_v
        #   2) 用户侧：user_embedding，承担论文 UCE s_u 的角色
        #   3) 物品协同侧：item_embedding，承担论文 COCE z_v 的角色
        # [注意] 代码变量名并没有严格照搬论文符号，判断对应关系要看“它是怎么得到/怎么使用的”。
        # ============================================================
        self.args = args

        # [可学习参数][内容侧]
        # 所有 category/attribute 的表示矩阵。每一行可以理解为一个属性的 embedding。
        # 后面的 forward 会对当前物品拥有的属性做 attention 加权，得到 final_attr_emb。
        self.attr_matrix = torch.nn.Parameter(torch.FloatTensor(args.attr_num, args.attr_present_dim))

        # 定义属性attribute注意力层
        # [可学习参数][内容侧] 学习各 category/attribute 的重要性分数；对物品已有属性做 Attention 加权融合，最终得到属性侧表示 final_attr_emb（非最终 CBCE）
        self.attr_W1 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim, args.attr_present_dim))
        self.attr_b1 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim, 1))
        self.attr_W2 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim, 1))

        # 控制整个模型的激活函数
        self.h = nn.LeakyReLU()

        # 图像的映射矩阵
        # [可学习参数][内容侧] Amazon-VG 的原始图像特征是 4096 维，这里映射到 implicit_dim（默认 256）。
        self.image_projection = torch.nn.Parameter(torch.FloatTensor(4096, args.implicit_dim))
        self.sigmoid = torch.nn.Sigmoid()  # 将门控信号映射到[0, 1]之间

        # user和item的嵌入层，可用预训练的进行初始化
        # [论文对应]
        #   user_embedding[user]：承担论文 UCE s_u 的角色
        #   item_embedding[item]：承担论文 COCE z_v 的角色
        # 这两套 embedding 来自协同侧，训练时参与 BPR/对比学习；严格冷启动测试物品不能依赖 item_embedding。
        # [当前默认流程] pretrain=False：从零创建 user_embedding（UCE s_u）和 item_embedding（COCE z_v），随机初始化后随训练更新。
        if args.pretrain is True:
            if args.pretrain_update is True:
                self.user_embedding = nn.Parameter(torch.load('user_emb.pt'), requires_grad=True)
                self.item_embedding = nn.Parameter(torch.load('item_emb.pt'), requires_grad=True)
            else:
                self.user_embedding = nn.Parameter(torch.load('user_emb.pt'), requires_grad=False)
                self.item_embedding = nn.Parameter(torch.load('item_emb.pt'), requires_grad=False)
        else:
            # 默认流程在这里
            self.user_embedding = nn.Parameter(torch.FloatTensor(args.user_number, args.implicit_dim))
            self.item_embedding = nn.Parameter(torch.FloatTensor(args.item_number, args.implicit_dim))

        # 定义生成层：q_v_a（约对应 Figure 2 的内容表示 c_v）-> g_c -> q_v_c（对应 Figure 2 的 CBCE q_v）
        # [论文对应] 这两层 MLP 是 Figure 2 中 Content CF Module / CBCE encoder g_c 的核心实现。
        # [注意] 原注释称输入为(q_v_a, u)，但当前 forward 实际只输入 q_v_a（≈c_v），没有直接输入用户表示 s_u。
        self.gen_layer1 = nn.Linear(args.attr_present_dim*2, args.cat_implicit_dim)
        self.gen_layer2 = nn.Linear(args.attr_present_dim, args.attr_present_dim)

        # 参数初始化
        self.__init_param__()

        #你现在完全不需要深究 Xavier 的数学公式，只需要知道：
        #这些参数刚创建出来的时候总得先有一些数值，不能全都乱来，所以用一种常见的初始化方法给它们设置合理的随机初值。之后真正的训练再不断修改这些数值。
    def __init_param__(self):
        # [代码作用] Xavier 初始化只决定训练开始时参数的初值；这些参数之后仍会由反向传播继续学习。
        nn.init.xavier_normal_(self.attr_matrix)
        nn.init.xavier_normal_(self.attr_W1)
        nn.init.xavier_normal_(self.attr_W2)
        nn.init.xavier_normal_(self.attr_b1)
        nn.init.xavier_normal_(self.image_projection)
        # 生成层初始化
        # user, item嵌入层的初始化, 没有预训练的情况下就初始化
        if self.args.pretrain is False:
            nn.init.xavier_normal_(self.user_embedding)
            nn.init.xavier_normal_(self.item_embedding)
        nn.init.xavier_normal_(self.gen_layer1.weight)
        nn.init.xavier_normal_(self.gen_layer2.weight)

    def forward(self, attribute, image_feature, batch_size):
        # ============================================================
        # [阅读重点] 第 2 步：forward 只负责“内容 -> CBCE”。
        # 论文 Figure 2 的主线可粗略对应为：
        #   X_v（category + image）
        #        -> 内容融合表示（代码 q_v_a，大致承担 c_v 的角色）
        #        -> g_c（gen_layer1 + gen_layer2）
        #        -> q_v（代码 q_v_c，最终 CBCE）
        # L_c、L_z、L_q 不在 forward 里，而是在下面的 train() 中计算。
        # ============================================================

        # [！！！注意：不要和论文符号混淆！！！]
        # 这里局部变量虽然命名为 z_v，但它【不是】论文 Figure 2 中的 COCE z_v。
        # 它只是为所有 attribute/category 计算 attention 打分的中间变量。
        # 论文 COCE z_v 在这份代码中实际由 model.item_embedding[item] 承担。
        z_v = torch.matmul(torch.matmul(self.attr_matrix, self.attr_W1)+self.attr_b1.squeeze(), self.attr_W2)
        z_v_copy = z_v.repeat(batch_size, 1, 1)
        z_v_squeeze = z_v_copy.squeeze(dim=2).to(device)

        # [代码作用] attribute 中当前物品不存在的类别位置为 -1；把这些位置变成极小值，
        # 这样 softmax 后它们的 attention 权重就近似为 0。
        neg_inf = torch.full(z_v_squeeze.shape, -1e6).to(device)
        z_v_mask = torch.where(attribute != -1, z_v_squeeze, neg_inf)
        attr_attention_weight = torch.softmax(z_v_mask, dim=1)

        # [内容侧] 将当前物品拥有的多个 category/attribute embedding 按 attention 权重融合。
        # [Shape] 结果每个物品得到一个 attr_present_dim 维向量（默认 256）。
        final_attr_emb = torch.matmul(attr_attention_weight, self.attr_matrix)

        # [可暂时跳过] 这一行计算了归一化图像特征，但后续 p_v 使用的仍是 image_feature 本身。
        image_norm = torch.nn.functional.normalize(image_feature, dim=1)

        # [内容侧] 4096维图像特征 -> implicit_dim维图像 embedding（默认 256）。
        p_v = torch.matmul(image_feature, self.image_projection)  # item的图像嵌入向量

        # [论文对应] 将属性表示与图像表示拼接，形成多模态内容表示。
        # q_v_a 是代码自己的中间变量名，不是论文正式符号；从数据流位置看，大致承担论文 c_v 的角色。
        # [Shape] 默认 256 + 256 = 512 维。
        q_v_a = torch.cat((final_attr_emb, p_v), dim=1)

        # [论文对应] gen_layer1 + LeakyReLU + gen_layer2 ≈ CBCE encoder g_c。
        # [论文符号] q_v_c 对应论文最终 CBCE q_v：只靠内容就能生成，因此冷物品测试时仍可获得。
        q_v_c = self.gen_layer2(self.h(self.gen_layer1(q_v_a)))
        return q_v_c


def train(model, train_loader, optimizer, valida, args, model_save_dir):
    # ============================================================
    # [阅读重点] 第 4/5 步：train() 是理解论文三个 Loss 的核心。
    # 每个 batch 的主流程：
    #   A. 内容 -> q_v_c（Figure 2：CBCE q_v）
    #   B. q_v_c（q_v）与 item_embedding（COCE z_v）做对比学习（L_c）
    #   C. item_embedding（z_v）+ user_embedding（UCE s_u）做 BPR（L_z）
    #   D. q_v_c（q_v）+ user_embedding（s_u）再做 BPR（L_q）
    #   E. 合成 total_loss -> backward() -> optimizer.step()
    # ============================================================
    print("model start train!")
    test_save_path = model_save_dir + "/result.csv"
    print("model train at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # 写入超参数
    with open(model_save_dir + "/readme.txt", 'a+') as f:
        str_ = args_tostring(args)
        f.write(str_)
        f.write('\nsave dir:'+model_save_dir)
        f.write('\nmodel train time:'+(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    with open(test_save_path, 'a+') as f:
        f.write("loss,contrast_sum,hr@5,hr@10,hr@20,ndcg@5,ndcg@10,ndcg@20\n")
    save_index = 0
    for i_epoch in range(args.epoch):
        i_batch = 0
        batch_time = time.time()
        # user                当前正用户
        # item                当前物品
        #
        # item_genres         当前物品的 category/attribute
        # item_img_feature    当前物品的图像特征
        #
        # neg_user            BPR 用的负用户
        #
        # positive_item_list  对比学习用的正物品
        # negative_item_list  对比学习用的负物品
        #
        # self_neg_list       self contrast 用的负物品
        for user, item, item_genres, item_img_feature, neg_user, positive_item_list, negative_item_list, self_neg_list in tqdm(train_loader):
            # [代码作用] 清空上一批次的梯度。
            optimizer.zero_grad()
            model.train()
            # allocate memory cpu to gpu
            model = model.to(device)
            user = user.to(device)
            item = item.to(device)
            item_genres = item_genres.to(device)
            item_img_feature = item_img_feature.to(device)
            neg_user = neg_user.to(device)
            positive_item_list = positive_item_list.to(device)
            negative_item_list = negative_item_list.to(device)

            # ========================================================
            # [训练阶段 A] Content -> q_v_c（Figure 2：CBCE q_v）
            # ========================================================
            # 调用 forward()，由当前物品的类别和图像侧信息生成 q_v_c（论文 CBCE q_v）。
            q_v_c = model(item_genres, item_img_feature, user.shape[0])
            q_v_c_unsqueeze = q_v_c.unsqueeze(dim=1)

            # ========================================================
            # [训练阶段 B] Contrastive Learning：q_v（q_v_c）↔ COCE z_v（item_embedding），计算 L_c
            # ========================================================
            # [实现对应] 论文写作 v -> g_v -> z_v；当前代码直接用可学习的 item_embedding 承担 COCE z_v 的角色。
            # 正样本：取与当前物品具有协同关联的物品，其 item_embedding 对应论文 z_{v+}。
            positive_item_emb = model.item_embedding[positive_item_list]

            # 计算 q_v 与正物品 z_{v+} 的余弦相似度 / tau，希望二者更接近。
            pos_contrast_mul = torch.sum(torch.mul(q_v_c_unsqueeze, positive_item_emb), dim=2) / (
                    args.tau * torch.norm(q_v_c_unsqueeze, dim=2) * torch.norm(positive_item_emb, dim=2))
            pos_contrast_exp = torch.exp(pos_contrast_mul)  # shape = 1024*10

            # 负样本：取负物品的 COCE z_{v-}，用于让 q_v 与这些协同不相关物品拉远。
            neg_item_emb = model.item_embedding[negative_item_list]
            q_v_c_un2squeeze = q_v_c_unsqueeze.unsqueeze(dim=1)

            # 计算 q_v 与负物品 z_{v-} 的余弦相似度 / tau。
            neg_contrast_mul = torch.sum(torch.mul(q_v_c_un2squeeze, neg_item_emb), dim=3) / (
                    args.tau * torch.norm(q_v_c_un2squeeze, dim=3) * torch.norm(neg_item_emb, dim=3))
            neg_contrast_exp = torch.exp(neg_contrast_mul)
            neg_contrast_sum = torch.sum(neg_contrast_exp, dim=2)  # shape = [1024, 10]

            # InfoNCE：让 q_v 更接近正物品 z_{v+}，并远离负物品 z_{v-}。
            contrast_val = -torch.log(pos_contrast_exp / (pos_contrast_exp + neg_contrast_sum))  # shape = [1024*10]
            contrast_examples_num = contrast_val.shape[0] * contrast_val.shape[1]
            contrast_sum = torch.sum(torch.sum(contrast_val, dim=1), dim=0) / contrast_val.shape[1]  # 同一个batch求mean

            '''
            contrast self
            '''
            # [代码额外实现] 除 q_v ↔ z_{v+} 外，还让 q_v 与当前物品自身的 COCE z_v 拉近，并与 self_neg_list 中负物品拉远。
            self_neg_item_emb = model.item_embedding[self_neg_list]
            self_neg_contrast_mul = torch.sum(torch.mul(q_v_c_unsqueeze, self_neg_item_emb), dim=2)/(
                args.tau*torch.norm(q_v_c_unsqueeze, dim=2)*torch.norm(self_neg_item_emb, dim=2))
            self_neg_contrast_sum = torch.sum(torch.exp(self_neg_contrast_mul), dim=1)
            item_emb = model.item_embedding[item]
            self_pos_contrast_mul = torch.sum(torch.mul(q_v_c, item_emb), dim=1) / (
                    args.tau * torch.norm(q_v_c, dim=1) * torch.norm(item_emb, dim=1))
            self_pos_contrast_exp = torch.exp(self_pos_contrast_mul)  # shape = 1024*1
            self_contrast_val = -torch.log(self_pos_contrast_exp/(self_pos_contrast_exp+self_neg_contrast_sum))
            self_contrast_sum = torch.sum(self_contrast_val)

            # ========================================================
            # [训练阶段 C] COCE 路径：z_v（item_emb）+ s_u（user_emb）-> f_z -> BPR L_z
            # ========================================================
            # 对当前物品 z_v，分别取交互过它的正用户 s_{u+} 和负用户 s_{u-}，要求正用户得分更高。
            user_emb = model.user_embedding[user]
            item_emb = model.item_embedding[item]
            neg_user_emb = model.user_embedding[neg_user]
            logsigmoid = torch.nn.LogSigmoid()

            # f_z：分别计算 z_v 与正/负用户的内积推荐分数。
            y_uv = torch.mul(item_emb, user_emb).sum(dim=1)
            y_kv = torch.mul(item_emb, neg_user_emb).sum(dim=1)

            # L_z：BPR 要求正用户分数 y_uv > 负用户分数 y_kv。
            y_ukv = -logsigmoid(y_uv - y_kv).sum()

            # ========================================================
            # [训练阶段 D] CBCE 路径：q_v（q_v_c）+ s_u（user_emb）-> f_q -> BPR L_q
            # ========================================================
            # 与上面的 L_z 相同，只是把协同表示 z_v 换成由侧信息生成的 q_v。
            y_uv2 = torch.mul(q_v_c, user_emb).sum(dim=1)
            y_kv2 = torch.mul(q_v_c, neg_user_emb).sum(dim=1)

            # L_q：同样要求正用户对 q_v 的得分高于负用户。
            y_ukv2 = -logsigmoid(y_uv2 - y_kv2).sum()

            # ========================================================
            # [训练阶段 E] 合并总 Loss
            # ========================================================
            # 对比学习 L_c 与两条 BPR 路径 L_z、L_q 联合训练；lambda1 控制两部分权重。
            # 注意官方代码采用 lambda1 与 (1-lambda1) 的加权形式，阅读时以实际代码为准。
            total_loss = args.lambda1*(contrast_sum+self_contrast_sum) + (1-args.lambda1)*(y_ukv+y_ukv2)
            if math.isnan(total_loss):
                print("loss is nan!, exit.", total_loss)
                exit(255)

            # ========================================================
            # [训练阶段 F] 真正发生“学习”的两行
            # ========================================================
            # backward()：根据 total_loss 计算所有 requires_grad=True 参数的梯度。
            # optimizer.step()：根据梯度更新 attr/image/g_c/user_embedding/item_embedding 等可学习参数。
            total_loss.backward()
            optimizer.step()
            i_batch += 1
            if i_batch % args.save_batch_time == 0:
                model.eval()
                print("[{},/13931603]total_loss:,{},{},s".format(i_batch*1024, total_loss.item(), int(time.time()-batch_time)))
                with torch.no_grad():
                    hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20 = valida.start_validate(model)
                with open(test_save_path, 'a+') as f:
                    f.write("{},{},{},{},{},{},{},{}\n".format(total_loss.item(), contrast_sum, hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20))
                # 保存模型
                batch_time = time.time()
                save_index += 1
                torch.save(model.state_dict(), model_save_dir + '/' + str(save_index)+".pt")


if __name__ == '__main__':
    # ============================================================
    # [阅读重点] 第 0 步/入口：这一段负责把数据、模型、优化器和验证器组装起来，然后调用 train()。
    # 第一次读模型原理时可以先略读；当你想弄清“程序究竟从哪里启动”时再回来细看。
    # ============================================================

    # result save dir
    save_dir = 'result/' + time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time()))
    os.makedirs(save_dir)
    # args
    args = get_args()
    print("progress start at:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    train_path = "data/train_withneg_rating.csv"
    vliad_path = 'data/validate_rating.csv'
    train_df = pd.read_csv(train_path)
    total_user_set = train_df['reviewerID']
    user_ser_dict = serialize_user(total_user_set)
    asin_category_int_map, category_ser_map = serial_asin_category()
    img_feature_dict = get_img_feature_pickle()
    # write internal variable
    with open(save_dir+"/save_dict.pkl", "wb") as file:
        save_dict = {'img_feature_dict': img_feature_dict, 'asin_category_int_map': asin_category_int_map,
                     'category_ser_map_len': category_ser_map.__len__(), 'user_ser_dict': user_ser_dict}
        pickle.dump(save_dict, file)

    # load dataset
    # [论文对应] RatingDataset 不只是“读取数据”，还会在 __getitem__ 中准备 BPR 负用户、
    # 对比学习正物品/负物品。理解采样方式时应跳到 support.py 的 RatingDataset.__getitem__。
    dataSet = RatingDataset(train_df, img_feature_dict, asin_category_int_map, category_ser_map.__len__(),
                            user_ser_dict, args.positive_number, args.negative_number)
    args.user_number = dataSet.user_number
    args.item_number = dataSet.item_number
    train_loader = torch.utils.data.DataLoader(dataSet, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print("模型超参数:", args_tostring(args))
    myModel = CCFCRec(args)

    # [论文对应] weight_decay 是代码层面的参数正则化手段之一，对应我们阅读论文总目标时讨论的参数约束思想。
    optimizer = torch.optim.Adam(myModel.parameters(), lr=args.learning_rate, weight_decay=0.1)
    validator = Validate(validate_csv=vliad_path, user_serialize_dict=user_ser_dict, img=img_feature_dict,
                         genres=asin_category_int_map, category_num=category_ser_map.__len__())
    train(myModel, train_loader, optimizer, validator, args, save_dir)