from collections import Counter
import numpy as np
import pandas as pd

# Константы окна (можно также передавать как параметры)
SAMPLING_RATE_HZ = 100
SENSOR_RESPONSE_TIME_SEC = 30
SENSOR_RECOVERY_TIME_SEC = 30
WINDOW_LEN_SEC = SENSOR_RESPONSE_TIME_SEC + SENSOR_RECOVERY_TIME_SEC
WINDOW_SIZE = WINDOW_LEN_SEC * SAMPLING_RATE_HZ
STEP_SIZE = int(WINDOW_SIZE * (1 - 0.8))  # 80% перекрытие
LABEL_THRESHOLD = 0.95  # Порог чистоты окна по основному классу

def create_windows_from_range(df, start_idx, end_idx, step_size, label_threshold=LABEL_THRESHOLD):
    """
    Создаёт окна из временного ряда с метками.
    Возвращает:
      - список окон с данными сенсоров (np.array),
      - список меток для окон,
      - список информации о каждом окне (начало, конец, метка).
    """
    sensor_columns = [col for col in df.columns if 'Sensor' in col]
    windowed_data = []
    windowed_labels = []
    window_info = []

    actual_end = min(end_idx, len(df))
    total_windows = 0
    count_main = 0
    count_transitions = 0

    for start in range(start_idx, actual_end - WINDOW_SIZE + 1, step_size):
        end = start + WINDOW_SIZE
        if end > actual_end:
            break

        total_windows += 1
        window = df.iloc[start:end]
        sensor_data = window[sensor_columns].values.astype(np.float32)
        label_counts = Counter(window['label'])
        most_common_label, most_common_count = label_counts.most_common(1)[0]

        if most_common_count / WINDOW_SIZE >= label_threshold:
            label = most_common_label
            count_main += 1
        else:
            start_label = window['label'].iloc[0]
            end_label = window['label'].iloc[-1]

            if start_label == end_label:
                label = start_label
                count_main += 1
            else:
                label = f"{start_label}->{end_label}"
                count_transitions += 1

        windowed_data.append(sensor_data)
        windowed_labels.append(label)
        window_info.append({'start': start, 'end': end, 'label': label})

    print(f"- Всего возможных окон: {total_windows:,}")
    print(f"- Чистых окон (≥ {label_threshold*100:.0f}% одного класса): {count_main:,}")
    print(f"- Переходных окон: {count_transitions:,}")
    if total_windows > 0:
        print(f"- Процент переходных: {count_transitions / total_windows * 100:.2f}%")

    return windowed_data, windowed_labels, window_info

def split_and_create_windows(df, train_ratio=0.7, step_size=STEP_SIZE, label_threshold=LABEL_THRESHOLD):
    """
    Разделяет данные на обучающую и тестовую части,
    создает окна для каждой части.
    Возвращает: X_train, y_train, X_test, y_test
    """
    total_len = len(df)
    test_start_idx = int(total_len * train_ratio)
    train_end_idx = test_start_idx

    print(f"Разделение данных:")
    print(f"- Для обучения/валидации: строки 0 - {train_end_idx:,}")
    print(f"- Для тестирования: строки {test_start_idx:,} - {total_len:,}")

    # Тестовая выборка с шагом в половину STEP_SIZE для лучшей детализации
    df_test_raw = df.iloc[test_start_idx:].copy().reset_index(drop=True)
    test_step = step_size // 2 if step_size > 1 else 1
    print(f"\nСоздаем тестовые окна с шагом {test_step}...")
    test_windows, test_labels, test_info = create_windows_from_range(df_test_raw, 0, len(df_test_raw), test_step, label_threshold)

    X_test = np.stack(test_windows)
    y_test = np.array(test_labels)

    print("\nРаспределение классов в тестовой выборке:")
    print(pd.Series(y_test).value_counts().sort_index())

    # Обучающая выборка
    df_train_raw = df.iloc[:train_end_idx].copy()
    print(f"\nСоздаем обучающие окна с шагом {step_size}...")
    train_windows, train_labels, train_info = create_windows_from_range(df_train_raw, 0, len(df_train_raw), step_size, label_threshold)

    X_train = np.stack(train_windows)
    y_train = np.array(train_labels)

    print("\nРаспределение классов в обучающей выборке:")
    print(pd.Series(y_train).value_counts().sort_index())

    return X_train, y_train, train_info, X_test, y_test, test_info
