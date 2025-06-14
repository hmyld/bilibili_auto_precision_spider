# bilibili_auto_precision_spider
这是一个基于计算机视觉对哔哩哔哩图文栏目标进行精准爬取的脚本。仅供视觉训练学习使用，严禁商用！！！！！
Bilibili 图片爬虫与预处理工具
## 简介
这是一个功能强大的 Bilibili 图片爬虫与预处理工具，可帮助用户从 Bilibili 网站抓取特定关键词的图片，并对这些图片进行格式转换、去重、降噪等一系列预处理操作。整个工具采用模块化设计，各个功能可以独立运行，也可以通过一键启动功能按顺序执行多个处理步骤。本脚本有一种创新性的基于视觉模型处理图片的方法，需要一个符合脚本的视觉模型。如果您不会写，我提供了一份AI生成的计算机视觉代码，model.py可供使用。
## 目录
- 功能特性
- 环境要求
- 安装步骤
- 使用说明
- 参数配置
- 功能模块详解
- 注意事项
- 贡献与反馈
## 1. 功能特性
### 爬虫功能
基于 Playwright 框架，可自动爬取 Bilibili 图文内容
支持多关键词搜索，可指定爬取页数
自动处理 cookies，避免频繁登录
智能识别并下载图片资源
### 图片预处理功能
- 格式转换：将爬取的 WebP 格式图片批量转换为 PNG 格式
- 去重处理：通过感知哈希算法识别并删除重复图片，支持自定义相似度阈值
- 小文件清理：自动删除小于指定大小的图片文件
- 图片降噪：提供多种降噪算法（高斯滤波、阈值法、K-means 聚类），有效提升图片质量
- 智能筛选：基于深度学习模型，可根据图片内容进行智能筛选
- 自动化执行
提供一键启动功能，可按需求组合执行多个处理步骤
有详细的日志输出，便于跟踪处理进度
## 2. 环境要求
Python 版本：3.8+（推荐 3.9 及以上版本），原生Python版本为3.12.7
依赖库：
```plaintext
playwright
requests
Pillow
imageio
opencv-python
scikit-learn
tensorflow
浏览器驱动：需要安装 Playwright 的 Chromium 浏览器驱动
深度学习模型：若使用智能筛选功能，需提前训练好相应的 TensorFlow 模型
```
## 3. 安装步骤
克隆或下载项目代码到本地
安装必要的依赖库：
bash
pip install playwright requests pillow imageio opencv-python scikit-learn tensorflow
python -m playwright install chromium
准备深度学习模型（如果需要使用智能筛选功能）
## 4. 使用说明
### 4.1 配置参数
在代码开头的参数配置部分，设置各项功能所需的参数，详情见后文。
### 4.2 执行单个功能
可以单独调用各个功能模块：
```python
# 启动爬虫
scrape_begin()

# 格式转换
convert_begin()

# 图片去重
remove_duplicate_begin()

# 删除过小图片
delete_small_begin()

# 图片降噪
denoise_begin()

# 智能筛选
Ose_preprocessing_traverse_begin()
```
### 4.3 一键执行多个功能
通过auto_execute函数组合执行多个功能：
```python
# 依次是 爬虫、格式转换、去重、删除小图片、降噪、预处理暴力降噪
# True表示执行，False表示不执行
customer_needs = [True, True, True, True, True, False]

if __name__ == "__main__":
    auto_execute(customer_needs)
```
## 5. 参数配置
### 5.1 爬虫参数
```
参数名	描述
target_charas	爬取目标关键词列表，例如：["动漫", "风景"]
pages	爬取的页数，一页约 500-1000 张图片
save_address	图片保存位置
now_cookies	B 站可用 cookies，需定期更新以避免被封禁
```
### 5.2 去重参数
```
参数名	描述
r_root_dir	要处理的根目录路径
r_dry_run	是否只显示结果不执行删除（True/False）
r_interactive	是否交互式确认删除（True/False）
r_min_size	最小文件大小 (KB)，小于此大小的文件会被忽略
r_threshold	汉明距离阈值，0 相当于只用 MD5，越大越宽松
```
### 5.3 其他参数
其他功能模块的参数配置类似，详细说明请参考代码中的注释部分。
## 6. 功能模块详解
### 6.1 Bilibili 爬虫
核心函数：scrape_bilibili和scrape_picture
工作流程：
搜索指定关键词的图文内容
解析搜索结果，提取文章链接
访问文章页面，下载其中的图片
注意事项：
频繁爬取可能导致 IP 被封禁，请合理设置爬取间隔
需要定期更新 cookies 以保持登录状态
### 6.2 格式转换
功能：将 WebP 格式图片批量转换为 PNG 格式
处理逻辑：
遍历指定目录下的所有文件
识别 WebP 格式文件并转换为 PNG
删除原始 WebP 文件
### 6.3 图片去重
算法：基于感知哈希（pHash）算法
处理流程：
计算所有图片的感知哈希值
对比哈希值，找出相似图片
根据文件大小等信息，选择保留最优图片，删除其他相似图片
### 6.4 小文件清理
功能：自动删除小于指定大小的图片文件
用途：清理爬虫过程中可能下载的无效图片
### 6.5 图片降噪
支持算法：
高斯滤波：平滑图像，减少噪点
阈值法：基于像素阈值进行二值化处理
K-means 聚类：基于颜色分布进行聚类降噪
处理流程：可按顺序组合执行多种降噪算法
### 6.6 智能筛选
原理：基于预训练的深度学习模型对图片进行分类筛选
用途：例如，从爬取的图片中筛选出特定主题的图片
## 7. 注意事项
- 爬虫风险：
频繁爬取可能导致 IP 被 B 站封禁，请合理设置爬取间隔和页数
需要定期更新 cookies 以保持登录状态
- 数据安全：
图片去重和小文件清理功能可能会删除重要文件，请在执行前备份数据
建议先在测试目录下验证功能，确认无误后再处理重要数据
- 性能考虑：
图片降噪和智能筛选功能可能需要较长时间，尤其是处理大量图片时
建议在配置较高的机器上运行
- 模型依赖：
智能筛选功能依赖预训练的深度学习模型，请确保模型路径配置正确
## 8.model.py程序使用说明
### 简介
本程序基于 TensorFlow 和 ResNet50 架构，实现了对动漫人物的图像识别功能。通过迁移学习和数据增强技术，程序能够准确识别特定动漫角色。程序支持模型训练、模型保存和新图像预测功能，并提供了命令行界面方便用户操作。
### 目录
- 环境要求
- 安装步骤
- 数据准备
- 参数配置
- 使用方法
- 训练模型
- 预测图像
- 训练历史可视化
- 注意事项
### 环境要求
Python 3.8+，原生Python环境为3.12.7
依赖库：
```plaintext
numpy
pandas
matplotlib
tensorflow
scikit-learn
argparse
pickle
```
### 安装步骤
- 安装 Python 环境（推荐 3.9 及以上版本）
- 安装所需依赖库：
```bash
pip install numpy pandas matplotlib tensorflow scikit-learn argparse pickle
```
- 准备预训练模型权重文件（ResNet50）
### 数据准备
- 组织训练数据目录结构，格式如下：
```plaintext
训练数据目录/
├── 类别1/
│   ├── 图像1.jpg
│   ├── 图像2.jpg
│   └── ...
├── 类别2/
│   ├── 图像3.jpg
│   ├── 图像4.jpg
│   └── ...
└── ...
```
- 每个类别目录下应包含足够数量的该类别的图像样本
- 确保图像质量良好，且主要内容为目标识别对象
### 参数配置
在代码开头的配置部分，可以修改以下参数：
```python
IMAGE_SIZE = (224, 224)  # 输入图像尺寸
BATCH_SIZE = 32  # 批次大小
EPOCHS = 50  # 训练轮次
TRAIN_DIR = r''  # 训练数据路径
MODEL_PATH = r'example.h5'  # 模型保存路径
PREPROCESSING_PARAMS_PATH = r'example.pkl'  # 预处理参数保存路径
```
### 使用方法
- 训练模型
使用以下命令训练模型：
```bash
python model.py --train
```
训练过程分为两个阶段：
第一阶段：冻结预训练模型的所有层，只训练自定义分类器
第二阶段：解冻预训练模型的最后几层，进行微调
训练过程中会自动保存验证集上表现最好的模型，并生成训练历史可视化图表。
- 预测图像
使用以下命令对单张图像进行预测：
```bash
python model.py --predict 图像路径
```
预测结果会显示识别的动漫人物类别及其置信度，同时还会输出完整的预测分布。
- 训练历史可视化
训练完成后，程序会自动生成训练历史图表，展示训练过程中准确率和损失的变化情况。图表会保存为training_history.png文件。
- 注意事项
1.训练数据的质量和数量直接影响模型的性能，请确保每个类别有足够多样本
2.预训练模型权重文件路径需要正确配置
3.预测时使用的图像尺寸会自动调整为模型训练时的尺寸
4.如果预测结果置信度低于 20%，程序会提示预测失败信息
5.程序使用了类别权重来处理数据不平衡问题，但仍建议尽量保持各类别样本数量相对均衡

## 9.贡献与反馈
  - **问题反馈**：如需报告 bug 或提出建议，请在GitHub Issues提交。
  - **代码贡献**：欢迎提交 PR 优化代码，建议先创建 Issue 讨论方案。
  - **支持作者**：用 QQ 1454988406 联系作者，如果您能赏给作者一瓶百事可乐，作者会很开心，并有继续改进代码的动力。
  - **开发者**：Ose Chen - SCUPI
  - **邮箱**：1454988406@qq.com
  - **项目地址**：GitHub [仓库链接](https://github.com/hmyld/bilibili_auto_precision_spider)

## 10.实验数据
```
对于五维向量，进行清洗测试，数据如下

    训练一个识别bangdream企划中乐队pastel*palette五名成员 丸山彩 冰川日菜 白鹭千圣 大和麻弥 若宫伊芙 的计算机视觉模型

    说明：第二次训练图片来源于人工筛选和第一次清洗结果。由于每次爬虫都是爬取相同网页，所以后续清洗会出现某些训练图片数量数据增长很多而某些几乎无增长

    组名    首次训练图片    首次模型测试数量    首次模型测试准确数量    首次测试模型准确度    首次清洗结果数量    首次清洗准确度    第二次训练图片    第二次清洗结果数量    第二次清洗准确数量    第二次清洗准确率    第二次模型测试数量    第二次模型测试准确数量    第二次模型测试准确度
   丸山彩      198                74                   29                39.19%             统计失败*         统计失败           395                31                   21                67.74%               74                      7                   9.46%
  冰川日菜     269               256                  112                43.75%                                                 339               102                   98                96.08%              256                    197                  76.95%
  白鹭千圣     487               463                  337                72.79%                                                 531               148                  148               100.00%              463                    155                  33.48%
  大和麻弥     367               353                  102                28.90%                                                 416                99                   85                85.86%              353                     86                  24.36%
  若宫伊芙     382               172                   52                30.23%                                                 488               278*                 199                71.58%              172                     86                  50.00%
                                                                                        *注：忘记统计                                            *注：这是白毛，导致一堆类似角色因为图片颜色/构图/灯光/背景等问题无法识别。人眼也难识别。
    数据说明：首次训练图片手动降噪。第0组为少数据训练组，第1组为特征明显组，第2组为大量数据组，第3组为对照组，第4组为特征不明显组。第二次训练图片为第一次加第一次筛选图片合并去重。
    对于特征明显的实验组，运行该方法进行训练数据的获取与清洗再次训练可以提升模型准确度。
    对于已经接近过拟合的实验组，该方法将会变成一个高效准确的爬虫
```
