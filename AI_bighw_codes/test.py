
import os
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 更新后的垃圾分类类别（与文件目录结构严格对应）
CLASS_NAMES = [
    'battery',       # 电池
    'biological',    # 生物垃圾
    'brown-glass',   # 棕色玻璃
    'cardboard',     # 纸板
    'clothes',       # 衣物
    'green-glass',   # 绿色玻璃
    'metal',         # 金属
    'paper',         # 纸张
    'plastic',       # 塑料
    'shoes',         # 鞋子
    'trash',         # 其他垃圾
    'white-glass'    # 透明玻璃
]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = len(CLASS_NAMES)  # 自动适配类别数量

# 数据准备函数（自动识别类别）
def prepare_data(data_dir, batch_size=BATCH_SIZE):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # 自动从目录结构获取类别（确保目录名与CLASS_NAMES一致）
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        classes=CLASS_NAMES  # 显式指定确保顺序一致
    )

    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        classes=CLASS_NAMES
    )

    return train_generator, val_generator

# 增强模型结构（增加网络容量）
def build_model():
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(256, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')  # 输出层动态适配
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 更新训练函数
def train_model(data_dir, epochs=50):
    train_gen, val_gen = prepare_data(data_dir)
    model = build_model()
    
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        epochs=epochs,
        callbacks=[early_stop]
    )
    
    model.save("enhanced_garbage_classifier.h5")
    return model, history

# 更新Gradio交互界面
def create_interface():
    def predict_image(img):
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        model = load_model("enhanced_garbage_classifier.h5")
        pred = model.predict(img_array, verbose=0)[0]
        
        # 构建带中文注释的返回结果
        class_translation = {
            'battery': '电池',
            'biological': '生物垃圾',
            'brown-glass': '棕色玻璃',
            'cardboard': '纸板',
            'clothes': '衣物',
            'green-glass': '绿色玻璃',
            'metal': '金属',
            'paper': '纸张',
            'plastic': '塑料',
            'shoes': '鞋子',
            'trash': '其他垃圾',
            'white-glass': '透明玻璃'
        }
        return {f"{class_translation[CLASS_NAMES[i]]} ({CLASS_NAMES[i]})": float(pred[i]) 
                for i in range(NUM_CLASSES)}
    
    return gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil", label="上传垃圾图片"),
        outputs=gr.Label(num_top_classes=5),
        title="智能垃圾分类系统（增强版）",
        description="支持12类垃圾识别：电池、生物垃圾、各类玻璃、纸制品、纺织品等",
        examples=[os.path.join("example_images", f) for f in os.listdir("example_images")],
        allow_flagging="never"
    )

if __name__ == "__main__":
    # 训练模型（注意数据集路径需包含所有子类别文件夹）
    train_model(r"C:\Users\Huawei\Desktop\AI_bighw\garbage_classification_train", epochs=50)
    
    # 启动交互界面
    create_interface().launch()
