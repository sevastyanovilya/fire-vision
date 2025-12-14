# FireVision

Классификация изображений для детекции дыма и огня по снимкам с БПЛА.

## Задача

Трёхклассовая классификация аэрофотоснимков:
- `none` — нет видимых признаков дыма или огня
- `smoke` — присутствует дым без видимого пламени
- `fire` — видимое пламя (может сопровождаться дымом)

Правило разметки: если на изображении присутствуют и огонь, и дым, класс — `fire`.

## Результаты

| Метрика | Значение |
|---------|----------|
| CV Macro F1 | 0.9811 |
| Public LB | 0.98301 |
| Место | 9/15 |

## Стек

- PyTorch, torchvision, timm
- Image Classification, Transfer Learning
- Stratified K-Fold, class weighting

## Подход

Решение использует EfficientNet-B0 с весами noisy student и 5-fold стратифицированную кросс-валидацию. Дисбаланс классов компенсируется взвешенной кросс-энтропией.

Конфигурация обучения:
- Модель: `tf_efficientnet_b0_ns` (timm)
- Размер изображения: 256x256
- Batch size: 32
- Оптимизатор: AdamW (lr=1e-4, weight_decay=1e-5)
- Scheduler: CosineAnnealingLR
- Эпох: 5 на фолд

Аугментации:
- RandomResizedCrop (scale 0.8-1.0)
- RandomHorizontalFlip
- ColorJitter (brightness, contrast, saturation, hue)

Инференс: усреднение предсказаний всех 5 фолдов.

## Структура проекта

```
fire-vision/
├── train.py          # Пайплайн обучения и инференса
├── requirements.txt  # Зависимости
└── README.md
```

## Использование

Скачать датасет:

```bash
kaggle competitions download -c fire-vision
unzip fire-vision.zip -d data/
```

Запуск обучения:

```bash
python train.py
```

Результат: `submission.csv` с предсказаниями для тестовой выборки.

## Требования

- Python 3.10+
- PyTorch 2.0+
- GPU с поддержкой CUDA (обучение ~20 мин на фолд на T4)

Установка зависимостей:

```bash
pip install -r requirements.txt
```

## Возможные улучшения

- Более крупные модели (EfficientNet-B3/B4, ConvNeXt)
- Test-time augmentation
- Pseudo-labeling на тестовой выборке
- Дополнительные аугментации (CutMix, MixUp)
- Focal loss вместо weighted CE
