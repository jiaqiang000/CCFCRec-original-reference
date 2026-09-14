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
        # [可学习参数][内容侧] 用于为不同 category/attribute 计算 attention 分数。
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
        if args.pretrain is True:
            if args.pretrain_update is True:
                self.user_embedding = nn.Parameter(torch.load('user_emb.pt'), requires_grad=True)
                self.item_embedding = nn.Parameter(torch.load('item_emb.pt'), requires_grad=True)
            else:
                self.user_embedding = nn.Parameter(torch.load('user_emb.pt'), requires_grad=False)
                self.item_embedding = nn.Parameter(torch.load('item_emb.pt'), requires_grad=False)
        else:
            self.user_embedding = nn.Parameter(torch.FloatTensor(args.user_number, args.implicit_dim))
            self.item_embedding = nn.Parameter(torch.FloatTensor(args.item_number, args.implicit_dim))

        # 定义生成层，将(q_v_a, u)的信息，共同生成 q_v_c， 生成包含协同信息的item嵌入
        # [论文对应] 这两层 MLP 可以理解为 Figure 2 中 Content CF Module / CBCE encoder g_c 的核心实现。
        # [注意] 原注释提到“(q_v_a, u)”，但当前 forward 实际送入 gen_layer 的只有 q_v_a，并没有直接拼接 user embedding。
        self.gen_layer1 = nn.Linear(args.attr_present_dim*2, args.cat_implicit_dim)
        self.gen_layer2 = nn.Linear(args.attr_present_dim, args.attr_present_dim)

        # 参数初始化
        self.__init_param__()

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
    #   A. 内容 -> q_v_c（CBCE q_v）
    #   B. q_v_c 与协同 item embedding 做对比学习（L_c）
    #   C. item_embedding + user_embedding 做 BPR（L_z）
    #   D. q_v_c + user_embedding 再做 BPR（L_q）
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
            # [训练阶段 A] Content -> CBCE q_v
            # ========================================================
            # [论文符号] q_v_c ≈ q_v。
            q_v_c = model(item_genres, item_img_feature, user.shape[0])
            q_v_c_unsqueeze = q_v_c.unsqueeze(dim=1)

            # ========================================================
            # [训练阶段 B] Contrastive Learning：计算 L_c
            # ========================================================
            # compute contrast loss
            # [论文对应] positive_item_list 来自 support.py 的协同正物品采样。
            # item_embedding[...] 承担论文 COCE z_{v+} 的角色。
            positive_item_emb = model.item_embedding[positive_item_list]

            # [论文公式] cosine(q_v, z_{v+}) / tau。
            # 分子是点积，分母用两个向量的范数做归一化，所以这里计算的是余弦相似度，再除温度 tau。
            pos_contrast_mul = torch.sum(torch.mul(q_v_c_unsqueeze, positive_item_emb), dim=2) / (
                    args.tau * torch.norm(q_v_c_unsqueeze, dim=2) * torch.norm(positive_item_emb, dim=2))
            pos_contrast_exp = torch.exp(pos_contrast_mul)  # shape = 1024*10

            # negative samples
            # [论文对应] 负协同物品的 COCE z_{v-}。
            neg_item_emb = model.item_embedding[negative_item_list]
            q_v_c_un2squeeze = q_v_c_unsqueeze.unsqueeze(dim=1)

            # [论文公式] cosine(q_v, z_{v-}) / tau。
            neg_contrast_mul = torch.sum(torch.mul(q_v_c_un2squeeze, neg_item_emb), dim=3) / (
                    args.tau * torch.norm(q_v_c_un2squeeze, dim=3) * torch.norm(neg_item_emb, dim=3))
            neg_contrast_exp = torch.exp(neg_contrast_mul)
            neg_contrast_sum = torch.sum(neg_contrast_exp, dim=2)  # shape = [1024, 10]

            # [论文对应] InfoNCE 风格：让 q_v 更接近正物品 COCE，并远离对应负物品 COCE。
            contrast_val = -torch.log(pos_contrast_exp / (pos_contrast_exp + neg_contrast_sum))  # shape = [1024*10]
            contrast_examples_num = contrast_val.shape[0] * contrast_val.shape[1]
            contrast_sum = torch.sum(torch.sum(contrast_val, dim=1), dim=0) / contrast_val.shape[1]  # 同一个batch求mean

            '''
            contrast self
            '''
            # [代码实现细节] 除“协同正物品”对比外，官方实现还加入当前 item 自身的 COCE 作为正例：
            # q_v_c（内容生成的 CBCE）与 item_embedding[item]（当前物品 COCE）做 self contrast。
            # 这部分最终以 self_contrast_sum 加入 total_loss。
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
            # [训练阶段 C] COCE 路径的 BPR：L_z
            # ========================================================
            # rank loss
            # [论文符号]
            #   user_emb     ≈ s_{u+}（正用户 UCE）
            #   item_emb     ≈ z_v（当前物品 COCE）
            #   neg_user_emb ≈ s_{u-}（负用户 UCE）
            user_emb = model.user_embedding[user]
            item_emb = model.item_embedding[item]
            neg_user_emb = model.user_embedding[neg_user]
            logsigmoid = torch.nn.LogSigmoid()

            # [论文对应] f_z 使用内积作为预测器。
            # 正用户得分 y_uv = z_v · s_{u+}；负用户得分 y_kv = z_v · s_{u-}。
            y_uv = torch.mul(item_emb, user_emb).sum(dim=1)
            y_kv = torch.mul(item_emb, neg_user_emb).sum(dim=1)

            # [论文对应] BPR：希望正用户得分 > 负用户得分。
            # y_ukv 承担论文 L_z 的角色。
            y_ukv = -logsigmoid(y_uv - y_kv).sum()

            # ========================================================
            # [训练阶段 D] CBCE 路径的 BPR：L_q
            # ========================================================
            # 使用属性生成item嵌入，再做一个bpr排序
            # [论文对应] 与上面完全相同的 BPR 思想，只是把 COCE item_emb 换成 CBCE q_v_c。
            # f_q 同样使用内积作为预测器。
            y_uv2 = torch.mul(q_v_c, user_emb).sum(dim=1)
            y_kv2 = torch.mul(q_v_c, neg_user_emb).sum(dim=1)

            # [论文对应] y_ukv2 承担论文 L_q 的角色。
            y_ukv2 = -logsigmoid(y_uv2 - y_kv2).sum()

            # ========================================================
            # [训练阶段 E] 合并总 Loss
            # ========================================================
            # [代码实现] contrast_sum + self_contrast_sum 是对比学习部分；y_ukv + y_ukv2 是两条 BPR 路径。
            # lambda1 控制二者权重。注意这里官方代码写成 lambda1 与 (1-lambda1) 的加权形式，
            # 阅读时应以实际代码为准，不要只按论文公式字面猜实现。
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
