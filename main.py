
from sklearn.metrics import accuracy_score, classification_report
from src.data_loader import download_and_extract_gdrive
from src.preprocessing import load_and_preprocess_data
from src.windowing import split_and_create_windows  # разбиение на окна
from src.visualization import plot_windows, plot_all_sensors_separately # визуализация окон
from src.feature_extraction import extract_advanced_features  # извлечение признаков
from src.balancing_and_validating import apply_selective_smote, cross_validate_train  # балансировка и кросс-валидация
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    FILE_ID = '1DvaCuhkQUxJY7N5lnqwnOwHCRdZp6ket'
    DEST_FOLDER = 'data'
    ZIP_FILENAME = 'ethylene_CO.txt.zip'
    DATA_FILENAME = 'ethylene_CO.txt'

    # Шаг 1: загрузка датасета
    download_and_extract_gdrive(FILE_ID, DEST_FOLDER, filename=ZIP_FILENAME, unzip=True)
    filepath = os.path.join(DEST_FOLDER, DATA_FILENAME)
    df = load_and_preprocess_data(filepath)
    
    # Шаг 2: визуализация датасета
    class_names = ['Нет газа', 'Ethylene', 'CO', 'Смесь']
    plot_all_sensors_separately(df, downsample_factor=100, class_names=class_names)

    # Шаг 3: удаление 2000 строк (переходный процесс после включения)
    rows_to_remove = 2000
    df = df.iloc[rows_to_remove:].reset_index(drop=True)
    print(f"Размер после удаления первых {rows_to_remove} строк: {len(df)} строк")
    
    # Шаг 4: Разделение на выборки и формирование окон
    X_train, y_train, train_info_new, X_test, y_test, test_info_new = split_and_create_windows(df)
    print(f"\nРазмеры сформированных данных:")
    print(f" - Обучающая выборка: X_train = {X_train.shape}, y_train = {y_train.shape}")
    print(f" - Тестовая выборка: X_test = {X_test.shape}, y_test = {y_test.shape}")

    # Шаг 5: Визуализация окон
    sensor_col = [col for col in df.columns if 'Sensor' in col][0]  # берем первый датчик для примера
    time_start_sec = 6200  # левые границы времени визуализации
    time_end_sec = 7000
    n_windows_to_plot = 80
    SAMPLING_RATE_HZ = 100
    plot_windows(df, sensor_col, train_info_new, SAMPLING_RATE_HZ, time_start_sec, time_end_sec, n_windows_to_plot)

    # Шаг 6: Извлечение признаков
    print("Извлекаем признаки из обучающей выборки...")
    train_features = extract_advanced_features(X_train)
    print(f"Форма тренировочных признаков: {train_features.shape}")
    print("Извлекаем признаки из тестовой выборки...")
    test_features = extract_advanced_features(X_test)
    print(f"Форма тестовых признаков: {test_features.shape}")
   
    # Шаг 7: Применяем SMOTE для балансировки
    X_train_balanced, y_train_balanced = apply_selective_smote(
        train_features,
        y_train,
        min_samples=2,
        max_samples=15,
        target_multiplier=2
    )

    # Шаг 8: Кросс-валидацмя
    models, cv_scores = cross_validate_train(X_train_balanced, y_train_balanced, n_splits=5)

    # Шаг 9: Обучаем и оцениваем модель с кросс-валидацией
    print("\n=== Обучение финальной модели на всех сбалансированных обучающих данных ===")
    final_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    final_model.fit(X_train_balanced, y_train_balanced)
    
    # Шаг 10: Оценка на тестовой выборке
    print("\n=== Оценка на тестовой выборке ===")
    y_test_pred = final_model.predict(test_features)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Итоговая точность (Final Accuracy): {test_accuracy:.4f}")
    print("\nОтчет о классификации:")
    print(classification_report(y_test, y_test_pred, zero_division=0))

if __name__ == "__main__":
    main()
