import torch
from torch import nn
from torch.nn import functional as F


# class TextNet(nn.Module):
#     def __init__(self, y_dim, bit, norm=True, mid_num1=1024*8, mid_num2=1024*8, hiden_layer=2):
#         """
#         :param y_dim: dimension of tags
#         :param bit: bit number of the final binary code
#         """
#         super(TextNet, self).__init__()
#         self.module_name = "txt_model"
#
#         mid_num1 = mid_num1 if hiden_layer > 1 else bit
#         modules = [nn.Linear(y_dim, mid_num1)]
#         if hiden_layer >= 2:
#             modules += [nn.ReLU(inplace=True)]
#             pre_num = mid_num1
#             for i in range(hiden_layer - 2):
#                 if i == 0:
#                     modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True)]
#                 else:
#                     modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
#                 pre_num = mid_num2
#             modules += [nn.Linear(pre_num, bit)]
#         self.fc = nn.Sequential(*modules)
#         #self.apply(weights_init)
#         self.norm = norm
#
#     def forward(self, x):
#         out = self.fc(x).tanh()
#         if self.norm:
#             norm_x = torch.norm(out, dim=1, keepdim=True)
#             out = out / norm_x
#         return out

# class TextNet(nn.Module):
#     def __init__(self, y_dim, bit, num_classes, norm=True, mid_num1=1024 * 8, mid_num2=1024 * 8, hiden_layer=2):
#         """
#         :param y_dim: dimension of tags
#         :param bit: bit number of the final binary code
#         :param num_classes: number of classes for the classification task
#         """
#         super(TextNet, self).__init__()
#         self.module_name = "txt_model"
#
#         mid_num1 = mid_num1 if hiden_layer > 1 else bit
#         modules = [nn.Linear(y_dim, mid_num1)]
#         if hiden_layer >= 2:
#             modules += [nn.ReLU(inplace=True)]
#             pre_num = mid_num1
#             for i in range(hiden_layer - 2):
#                 if i == 0:
#                     modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True)]
#                 else:
#                     modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
#                 pre_num = mid_num2
#             modules += [nn.Linear(pre_num, bit)]
#         self.fc = nn.Sequential(*modules)
#
#         # Add a new fully connected layer for classification
#         self.fc_classification = nn.Linear(mid_num2, num_classes)
#
#         self.norm = norm
#
#     def forward(self, x):
#         for i, layer in enumerate(self.fc):
#             x = layer(x)
#             # save the output of the second last layer
#             if i == len(self.fc) - 2:
#                 out_classification = self.fc_classification(x)
#
#         out_hash = x.tanh()
#         if self.norm:
#             norm_x = torch.norm(out_hash, dim=1, keepdim=True)
#             out_hash = out_hash / norm_x
#
#         return out_hash, out_classification

#最终版
class TextNet(nn.Module):
    def __init__(self, y_dim, bit, num_classes, norm=True, mid_num1=1024 * 8, mid_num2=1024 * 8, hiden_layer=2, dropout_rate=0.1):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        :param num_classes: number of classes for the classification task
        :param dropout_rate: probability to dropout a neuron
        """
        super(TextNet, self).__init__()
        self.module_name = "txt_model"

        self.dropout = nn.Dropout(p=dropout_rate) # Define the dropout layer

        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        modules = [nn.Linear(y_dim, mid_num1)]
        if hiden_layer >= 2:
            modules += [nn.ReLU(inplace=True), self.dropout]  # Add dropout after ReLU
            pre_num = mid_num1
            for i in range(hiden_layer - 2):
                if i == 0:
                    modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True), self.dropout]  # Add dropout after ReLU
                else:
                    modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True), self.dropout]  # Add dropout after ReLU
                pre_num = mid_num2
            modules += [nn.Linear(pre_num, bit)]
        self.fc = nn.Sequential(*modules)

        # Add a new fully connected layer for classification
        self.fc_classification = nn.Linear(bit, num_classes)

        self.norm = norm

    def forward(self, x):
        for i, layer in enumerate(self.fc):
            x = layer(x)
            # save the output of the second last layer


        out_hash = x.tanh()
        out_classification = self.fc_classification(out_hash)
        if self.norm:
            norm_x = torch.norm(out_hash, dim=1, keepdim=True)
            out_hash = out_hash / norm_x

        return out_hash, out_classification

# # 画图板
# class TextNet(nn.Module):
#     def __init__(self, y_dim, bit, num_classes, norm=True, mid_num1=1024 * 8, mid_num2=1024 * 8, hiden_layer=2):
#         """
#         :param y_dim: dimension of tags
#         :param bit: bit number of the final binary code
#         :param num_classes: number of classes for the classification task
#         """
#         super(TextNet, self).__init__()
#         self.module_name = "txt_model"
#
#         mid_num1 = mid_num1 if hiden_layer > 1 else bit
#         modules = [nn.Linear(y_dim, mid_num1)]
#         if hiden_layer >= 2:
#             modules += [nn.ReLU(inplace=True)]
#             pre_num = mid_num1
#             for i in range(hiden_layer - 2):
#                 if i == 0:
#                     modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True)]
#                 else:
#                     modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
#                 pre_num = mid_num2
#             modules += [nn.Linear(pre_num, bit)]
#         self.fc = nn.Sequential(*modules)
#
#         # Add a new fully connected layer for classification
#         self.fc_classification = nn.Linear(mid_num2, 1024)
#
#         self.norm = norm
#
#     def forward(self, x):
#         for i, layer in enumerate(self.fc):
#             x = layer(x)
#             # save the output of the second last layer
#             if i == 1:
#                 out_classification = self.fc_classification(x)
#
#         out_hash = x.tanh()
#         if self.norm:
#             norm_x = torch.norm(out_hash, dim=1, keepdim=True)
#             out_hash = out_hash / norm_x
#
#         return out_hash, out_classification


