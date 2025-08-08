import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def apply_selective_smote(X_features, y_labels, min_samples=2, max_samples=15, target_multiplier=3):
    """
    Применяет SMOTE только к классам с количеством образцов от min_samples до max_samples.
    """
    class_counts = pd.Series(y_labels).value_counts()
    print(f"\nИсходное распределение классов:")
    for class_name, count in class_counts.sort_index().items():
        print(f"  {class_name}: {count}")

    classes_to_augment = []
    target_counts = {}
    median_count = int(class_counts.median())

    for class_name, count in class_counts.items():
        if min_samples <= count <= max_samples:
            classes_to_augment.append(class_name)
            target_count = min(count * target_multiplier, median_count)
            target_counts[class_name] = max(target_count, max_samples + 5)

    if not classes_to_augment:
        print("Нет классов для применения SMOTE.")
        return X_features, y_labels

    print(f"\nКлассы для увеличения с помощью SMOTE: {classes_to_augment}")
    print("Целевые количества после SMOTE:")
    for class_name in classes_to_augment:
        print(f"  {class_name}: {class_counts[class_name]} -> {target_counts[class_name]}")

    sampling_strategy = {}
    for class_name, count in class_counts.items():
        if class_name in classes_to_augment:
            sampling_strategy[class_name] = target_counts[class_name]
        else:
            sampling_strategy[class_name] = count

    try:
        min_class_size = min([class_counts[c] for c in classes_to_augment])
        k_neighbors = min(3, max(1, min_class_size - 1))
        print(f"Используем k_neighbors = {k_neighbors}")

        smote = SMOTE(
            random_state=42,
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors
        )
        X_resampled, y_resampled = smote.fit_resample(X_features, y_labels)

        resampled_counts = pd.Series(y_resampled).value_counts()
        print(f"\nРаспределение классов после SMOTE:")
        for class_name, count in resampled_counts.sort_index().items():
            original_count = class_counts.get(class_name, 0)
            change = count - original_count
            print(f"  {class_name}: {count} (+{change})")

        print(f"\nОбщее количество образцов: {len(y_labels)} -> {len(y_resampled)}")
        return X_resampled, y_resampled

    except Exception as e:
        print(f"Ошибка при применении SMOTE: {e}")
        print("Попробуем более консервативный подход...")

        try:
            very_small_classes = [c for c in classes_to_augment if class_counts[c] <= 6]
            if very_small_classes:
                conservative_strategy = {}
                for class_name, count in class_counts.items():
                    if class_name in very_small_classes:
                        conservative_strategy[class_name] = max(count * 2, 12)
                    else:
                        conservative_strategy[class_name] = count

                smote_conservative = SMOTE(
                    random_state=42,
                    sampling_strategy=conservative_strategy,
                    k_neighbors=1
                )
                X_resampled, y_resampled = smote_conservative.fit_resample(X_features, y_labels)
                print("Применен консервативный SMOTE.")
                return X_resampled, y_resampled
        except Exception as e2:
            print(f"Консервативный SMOTE также не удался: {e2}")

        print("Возвращаем исходные данные.")
        return X_features, y_labels


def cross_validate_train(X, y, n_splits=5, random_state=42):
    """
    Кросс-валидация с обучением RandomForest с balanced class weights.

    Возвращает список моделей и список точностей по фолдам.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_scores = []
    models = []

    print(f"\n=== Обучение и оценка с помощью {n_splits}-фолдовой кросс-валидации ===")

    for fold, (train_index, val_index) in enumerate(skf.split(X, y), start=1):
        print(f"\nОбучение фолда {fold}/{n_splits}...")

        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        rf_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf_model.fit(X_train_fold, y_train_fold)
        models.append(rf_model)

        y_val_pred = rf_model.predict(X_val_fold)
        score = accuracy_score(y_val_fold, y_val_pred)
        cv_scores.append(score)

        print(f"  Точность фолда {fold}: {score:.4f}")

    print("\n" + "-" * 30)
    print("Результаты кросс-валидации:")
    print(cv_scores)
    print(f"Средняя точность: {np.mean(cv_scores):.4f}")
    print(f"Стандартное отклонение: {np.std(cv_scores):.4f}")

    return models, cv_scores
