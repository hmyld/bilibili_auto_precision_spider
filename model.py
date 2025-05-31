import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
import os
import argparse
import pickle

# ==============
# 1. 配置参数
# ==============
IMAGE_SIZE = (224, 224)  # 输入图像尺寸
BATCH_SIZE = 32  # 批次大小
EPOCHS = 50  # 增加总训练轮次
TRAIN_DIR = r''  # 训练数据路径
MODEL_PATH = r''  # 模型保存路径
PREPROCESSING_PARAMS_PATH = r''  # 预处理参数保存路径

# ==============
# 2. 数据准备（增强和预处理 + 类别权重计算）
# ==============
def prepare_data():
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=[0.7, 1.3],
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        featurewise_center=True,
        featurewise_std_normalization=True
    )
    
    train_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    # 计算并保存预处理参数
    datagen.fit(train_generator[0][0])
    preprocessing_params = {
        'mean': datagen.mean,
        'std': datagen.std
    }
    with open(PREPROCESSING_PARAMS_PATH, 'wb') as f:
        pickle.dump(preprocessing_params, f)
    
    # 计算类别权重
    y_train = train_generator.classes  # 获取训练集类别标签
    class_weights = class_weight.compute_class_weight(
        'balanced',  # 平衡类别权重
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))  # 转换为字典格式
    
    return train_generator, val_generator, class_weights  # 返回类别权重

# ==============
# 3. 构建模型（迁移学习 + 优化结构）
# ==============
def build_model(num_classes):
    weights_path = r'C:\Users\DX110\Desktop\remote_model\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    base_model = applications.ResNet50(
        weights=weights_path,
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.6),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ==============
# 4. 训练模型（两阶段训练策略 + 类别权重）
# ==============
def train_model():
    print("准备数据...")
    train_generator, val_generator, class_weights = prepare_data()  # 解包类别权重
    
    print("构建模型...")
    model = build_model(train_generator.num_classes)
    
    print("第一阶段训练（冻结预训练层）...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # 新增：在fit中传入class_weight参数
    history1 = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=20,
        callbacks=callbacks,
        class_weight=class_weights  # 应用类别权重
    )
    
    print("第二阶段训练（解冻最后几层）...")
    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 第二阶段同样应用类别权重
    history2 = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=EPOCHS - 20,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # 合并训练历史
    history = {
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss']
    }
    
    plot_training_history(history)
    
    print(f"模型已保存至: {MODEL_PATH}")
    return model

# ==============
# 5. 绘制训练历史
# ==============
def plot_training_history(history):
    """绘制训练和验证的准确率与损失曲线"""
    plt.figure(figsize=(12, 4))
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# ==============
# 6. 预测新图像
# ==============
def predict_image(model, image_path, class_indices):
    """预测单张图像中的动漫人物"""
    # 加载预处理参数
    if os.path.exists(PREPROCESSING_PARAMS_PATH):
        with open(PREPROCESSING_PARAMS_PATH, 'rb') as f:
            preprocessing_params = pickle.load(f)
        mean = preprocessing_params['mean']
        std = preprocessing_params['std']
    else:
        print("警告: 未找到预处理参数，使用默认值")
        mean = None
        std = None
    
    # 加载并预处理图像
    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 添加批次维度
    img_array = img_array / 255.0  # 归一化
    
    # 应用训练时的标准化参数
    if mean is not None and std is not None:
        img_array = (img_array - mean) / (std + 1e-7)
    
    # 预测
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index] * 100
    
    # 获取类别名称
    class_names = list(class_indices.keys())
    predicted_class = class_names[class_index]
    
    # 打印完整的预测结果分布
    print("\n完整预测分布:")
    for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
        print(f"{i+1}. {class_name}: {prob*100:.2f}%")
    
    return predicted_class, confidence

# ==============
# 7. 主函数
# ==============
def main():
    """主函数：训练或加载模型并进行预测"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='识别程序')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--predict', type=str, help='预测图像路径')
    args = parser.parse_args()
    
    # 训练模型
    if args.train:
        model = train_model()
        _, val_generator, _ = prepare_data()
        class_indices = val_generator.class_indices
    else:
        # 加载已有模型
        if not os.path.exists(MODEL_PATH):
            print(f"错误：模型文件 {MODEL_PATH} 不存在，请先训练模型！")
            return
        
        print(f"加载模型: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # 获取类别索引
        train_generator, val_generator, _ = prepare_data()  # 正确解包
        class_indices = val_generator.class_indices
    
    # 预测图像
    if args.predict:
        if not os.path.exists(args.predict):
            print(f"错误：图像文件 {args.predict} 不存在！")
            return
        
        predicted_class, confidence = predict_image(model, args.predict, class_indices)
        if confidence<20:
            print("没有匹配选项！")
        print(f"\n最终预测结果: {predicted_class}，置信度: {confidence:.2f}%")

if __name__ == "__main__":
    main()
