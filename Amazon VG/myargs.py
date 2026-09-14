import argparse


def get_args():
    # ============================================================
    # [阅读重点] 这里集中定义训练超参数。
    # 第一遍不需要死记所有参数，重点关注：
    #   batch_size / learning_rate
    #   positive_number / negative_number：L_c 的采样规模
    #   implicit_dim：用户/物品协同表示维度
    #   tau：对比学习温度系数
    #   lambda1：对比损失与 BPR 损失之间的权重
    # ============================================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024, help="batch_size")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')

    # [参数正则化] 这里定义了 weight_decay 参数。
    # [注意] 当前 model.py 创建 Adam 时实际直接写死 weight_decay=0.1，未读取 args.weight_decay。
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')

    # [论文对应][L_c] 每个当前物品采样多少个协同正物品、每个正例配多少负物品。
    parser.add_argument('--positive_number', type=int, default=5, help='contrast positive number')
    parser.add_argument('--negative_number', type=int, default=40, help='contrast negative number')

    # [注意] 这是官方原代码的参数名，其中包含空格；当前主要训练逻辑实际使用的是 negative_number。
    # 这里只做提示，不修改官方实现。
    parser.add_argument('--self negative_number', type=int, default=40, help='contrast negative number')

    # [内容侧] Amazon-VG 中 attribute/category 总数及其 embedding 维度。
    parser.add_argument('--attr_num', type=int, default=3127, help='item attribute number')
    parser.add_argument('--attr_present_dim', type=int, default=256, help='the dimension of present')

    # [论文符号] user_embedding / item_embedding（UCE s_u / COCE z_v）的维度。
    parser.add_argument('--implicit_dim', type=int, default=256, help='the dimension of u/i present')

    # [内容侧] 生成 CBCE q_v 的 MLP 中间/目标维度配置。
    parser.add_argument('--cat_implicit_dim', type=int, default=256, help='the q_v_c dimension')

    # [运行时会覆盖] model.py 中创建 RatingDataset 后会用数据集实际数量覆盖这两个值。
    parser.add_argument('--user_number', type=int, default=138493, help='user number in training set')
    parser.add_argument('--item_number', type=int, default=16803, help='item number in training set')

    # [论文对应][L_c] temperature tau：控制 softmax/InfoNCE 相似度分布的尖锐程度。
    parser.add_argument('--tau', type=float, default=0.1, help='contrast loss temperature')

    # [论文对应] model.py 中实际总损失：
    # lambda1 * (contrast losses) + (1-lambda1) * (two BPR losses)。
    parser.add_argument('--lambda1', type=float, default=0.6, help='collaborative contrast loss weight')

    parser.add_argument('--epoch', type=int, default=100, help='training epoch')

    # [消融/预训练相关] 控制 user/item 协同 embedding 是否从预训练参数加载、加载后是否继续更新。
    parser.add_argument('--pretrain', type=bool, default=False, help='user/item embedding pre-training')
    parser.add_argument('--pretrain_update', type=bool, default=False, help='u/i pretrain embedding update')

    # [注意] 下面两个 flag 在当前 Amazon-VG model.py 主训练流程中没有直接控制相应分支；保留官方原样。
    parser.add_argument('--contrast_flag', type=bool, default=True, help='contrast job flag')
    parser.add_argument('--user_flag', type=bool, default=False, help='use user to q_v_c flag')

    # 每训练多少个 batch 进行一次验证并保存模型。
    parser.add_argument('--save_batch_time', type=int, default=300, help='every batch time save the model')
    args = parser.parse_args()
    return args


def args_tostring(args):
    # [代码作用] 将全部参数转成文本，供 model.py 写入 result/.../readme.txt 留存实验配置。
    str_ = ""
    for arg in vars(args):
        str_ += str(arg) + ":" + str(getattr(args, arg)) + "\n"
    return str_
