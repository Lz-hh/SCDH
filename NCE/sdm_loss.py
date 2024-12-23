import torch
import torch.nn as nn
import torch.nn.functional as F
def compute_sdm(image_features, text_features, labels, logit_scale, factor=0.1, epsilon=1e-8):
    """
    Similarity Distribution Matching for cross-modal hashing.
    """
    image_features = torch.cat(image_features)
    text_features = torch.cat(text_features)

    # Compute label similarity
    labels_similarity = torch.matmul(labels.float(), labels.t().float())
    labels_dist = labels_similarity ** (1 / factor)  # Avoid division by zero

    # Normalize the features
    image_norm = image_features / torch.clamp(image_features.norm(dim=1, keepdim=True), min=epsilon)
    text_norm = text_features / torch.clamp(text_features.norm(dim=1, keepdim=True), min=epsilon)

    # Compute cosine similarity between image and text features
    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    # Apply scaling factor
    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # Normalize the true matching distribution
    labels_distribute = labels_dist / torch.clamp(labels_dist.sum(dim=1, keepdim=True), min=epsilon)

    # Compute loss
    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.clamp(torch.log(labels_distribute + epsilon),
                                                                               min=epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.clamp(torch.log(labels_distribute + epsilon),
                                                                               min=epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

def compute_in_modal_sdm(image_features, labels, logit_scale, epsilon=1e-8):
    """
    In-Modal Similarity Distribution Matching for hashing.
    """
    image_features = torch.cat(image_features)

    # Compute label similarity
    labels_similarity = torch.matmul(labels.float(), labels.t().float())
    labels_dist = labels_similarity / torch.clamp(labels_similarity.max(), min=epsilon)  # Avoid division by zero

    # Normalize the features
    image_norm = image_features / torch.clamp(image_features.norm(dim=1, keepdim=True), min=epsilon)

    # Compute cosine similarity among image features
    i2i_cosine_theta = image_norm @ image_norm.t()

    # Apply scaling factor
    image_proj_image = logit_scale * i2i_cosine_theta

    # Normalize the true matching distribution
    labels_distribute = labels_dist / torch.clamp(labels_dist.sum(dim=1, keepdim=True), min=epsilon)

    # Compute loss
    i2i_pred = F.softmax(image_proj_image, dim=1)
    i2i_loss = i2i_pred * (F.log_softmax(image_proj_image, dim=1) - torch.clamp(torch.log(labels_distribute + epsilon), min=epsilon))

    loss = torch.mean(torch.sum(i2i_loss, dim=1))

    return loss

#
# def compute_in_modal_sdm_txt(image_features, labels, logit_scale, epsilon=1e-8):
#     """
#     In-Modal Similarity Distribution Matching for hashing.
#     """
#     image_features = torch.cat(image_features)
#
#     # Compute label similarity
#     labels_similarity = torch.matmul(labels.float(), labels.t().float())
#     labels_dist = labels_similarity / torch.clamp(labels_similarity.max(), min=epsilon)  # Avoid division by zero
#
#     # Normalize the features
#     image_norm = image_features / torch.clamp(image_features.norm(dim=1, keepdim=True), min=epsilon)
#
#     # Compute cosine similarity among image features
#     i2i_cosine_theta = image_norm @ image_norm.t()
#
#     # Apply scaling factor
#     image_proj_image = logit_scale * i2i_cosine_theta
#
#     # Normalize the true matching distribution
#     labels_distribute = labels_dist / torch.clamp(labels_dist.sum(dim=1, keepdim=True), min=epsilon)
#
#     # Compute loss
#     i2i_pred = F.softmax(image_proj_image, dim=1)
#     i2i_loss = i2i_pred * (F.log_softmax(image_proj_image, dim=1) - torch.clamp(torch.log(labels_distribute + epsilon), min=epsilon))
#
#     loss = torch.mean(torch.sum(i2i_loss, dim=1))
#
#     return loss
#
#
#
# 二值相似性矩阵
# def compute_sdm(image_features, text_features, labels, logit_scale, factor=0.3, epsilon=1e-8):
#     """
#     Similarity Distribution Matching for cross-modal hashing.
#     """
#     image_features = torch.cat(image_features)
#     text_features = torch.cat(text_features)
#
#     # 计算标签相似度矩阵
#     labels_similarity = torch.matmul(labels.float(), labels.t().float())
#     # 将标签相似度矩阵转换为二值矩阵
#     labels_dist = (labels_similarity > 0).float()
#
#     # 归一化特征向量
#     image_norm = image_features / torch.clamp(image_features.norm(dim=1, keepdim=True), min=epsilon)
#     text_norm = text_features / torch.clamp(text_features.norm(dim=1, keepdim=True), min=epsilon)
#
#     # 计算余弦相似度
#     t2i_cosine_theta = text_norm @ image_norm.t()
#     i2t_cosine_theta = t2i_cosine_theta.t()
#
#     # 应用缩放因子
#     text_proj_image = logit_scale * t2i_cosine_theta
#     image_proj_text = logit_scale * i2t_cosine_theta
#
#     # 归一化真实匹配分布
#     labels_distribute = labels_dist / torch.clamp(labels_dist.sum(dim=1, keepdim=True), min=epsilon)
#
#     # 计算损失函数
#     i2t_pred = F.softmax(image_proj_text, dim=1)
#     i2t_loss = i2t_pred * (
#         F.log_softmax(image_proj_text, dim=1) - torch.clamp(torch.log(labels_distribute + epsilon), min=epsilon)
#     )
#     t2i_pred = F.softmax(text_proj_image, dim=1)
#     t2i_loss = t2i_pred * (
#         F.log_softmax(text_proj_image, dim=1) - torch.clamp(torch.log(labels_distribute + epsilon), min=epsilon)
#     )
#
#     loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
#
#     return loss
