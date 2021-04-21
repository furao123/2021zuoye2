#!/usr/bin/env python
# coding: utf-8

# # 作业一：分类结果展示

# ![](https://ai-studio-static-online.cdn.bcebos.com/0350c6891423417f8100032ad9158757312bd7b6b9064153ac31b06979a699f3)
# 

# # 作业二

# ## 安装飞桨

# In[ ]:


get_ipython().system('pip install paddlex')


# ## 文件解压，分配路径

# In[ ]:


get_ipython().system('unzip -oq data/json.zip')
get_ipython().system('unzip -oq data/jpeg.zip')


# In[ ]:


get_ipython().system('paddlex --data_conversion --source labelme --to MSCOCO         --pics jpeg         --annotations json         --save_dir dataset_coco')


# In[ ]:


get_ipython().system('paddlex --split_dataset --format COCO --dataset_dir dataset_coco --val_value 0.2 --test_value 0.1')


# In[ ]:


import paddlex as pdx


# ## 输入数据转换与扩充

# In[ ]:


from paddlex.det import transforms

# 定义训练和验证时的transforms
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(), transforms.Normalize(),
    transforms.ResizeByShort(
        short_size=800, max_size=1333), transforms.Padding(coarsest_stride=32)
])

eval_transforms = transforms.Compose([
    transforms.Normalize(), transforms.ResizeByShort(
        short_size=800, max_size=1333), transforms.Padding(coarsest_stride=32)
])


# 定义训练和验证所用数据集

# In[ ]:


#定义训练和验证所用数据集
train_dataset = pdx.datasets.CocoDetection(
    data_dir='dataset_coco/JPEGImages',
    ann_file='dataset_coco/train.json',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.CocoDetection(
    data_dir='dataset_coco/JPEGImages',
    ann_file='dataset_coco/val.json',
    transforms=eval_transforms)


# ## 初始化模型并训练

# In[ ]:


# 初始化模型并训练
num_classes = len(train_dataset.labels) + 1

model = pdx.det.MaskRCNN(num_classes=num_classes, backbone='ResNet50')

model.train(
    num_epochs=100,
    train_dataset=train_dataset,
    train_batch_size=4,
    eval_dataset=eval_dataset,
    learning_rate=0.00125,
    warmup_steps=10,
    lr_decay_epochs=[8, 11],
    save_dir='output/mask_rcnn_r50_fpn',
    use_vdl=True)


# ## 对图片进行预测

# In[ ]:


model = pdx.load_model('output/mask_rcnn_r50_fpn/best_model')
image_name = 'dataset_coco/JPEGImages/lhgd227.jpg'
result = model.predict(image_name)
pdx.det.visualize(image_name, result, threshold=0.5) 


# ![](https://ai-studio-static-online.cdn.bcebos.com/0238f26a734c42b28d17fe89bbb77802c656e2a9d51f414488ec23f3fc3410fc)
# 
