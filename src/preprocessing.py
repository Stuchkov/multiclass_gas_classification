import pandas as pd

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Загружает и предобрабатывает датасет из файла ethylene_CO.txt.
    - Считывает заголовок
    - Назначает имена столбцов для данных с датчиков
    - Назначает метки классов (label)
    - Удаляет столбцы с концентрациями газа и временем
    
    Возвращает Pandas DataFrame с предобработанными данными.
    """
    with open(filepath, 'r') as f:
        header = f.readline().strip().split(', ')

    sensor_order = ['TGS2602', 'TGS2602', 'TGS2600', 'TGS2600',
                    'TGS2610', 'TGS2610', 'TGS2620', 'TGS2620',
                    'TGS2602', 'TGS2602', 'TGS2600', 'TGS2600',
                    'TGS2610', 'TGS2610', 'TGS2620', 'TGS2620']

    column_names = header[:3] + [f'Sensor{i}_{sensor}' for i, sensor in enumerate(sensor_order, 1)]

    # Считываем данные с нужными именами колонок
    df = pd.read_csv(filepath,
                     delimiter='\s+',
                     skiprows=1,
                     header=None,
                     names=column_names)

    # Назначаем метки
    df['label'] = 0  # Без газа (фон)
    df.loc[(df['Ethylene conc (ppm)'] > 0) & (df['CO conc (ppm)'] == 0), 'label'] = 1  # Этилен
    df.loc[(df['Ethylene conc (ppm)'] == 0) & (df['CO conc (ppm)'] > 0), 'label'] = 2  # CO
    df.loc[(df['Ethylene conc (ppm)'] > 0) & (df['CO conc (ppm)'] > 0), 'label'] = 3  # Смесь

    # Для вывода примеров по одному для каждого label
    examples = pd.concat([
        df[df['label'] == 0].head(1),
        df[df['label'] == 1].head(1),
        df[df['label'] == 2].head(1),
        df[df['label'] == 3].head(1),
    ])

    print("Столбцы в DataFrame:", df.columns.tolist())
    print("\nПример по одному для каждого значения label (0–3):")
    print(examples[['Time (seconds)', 'CO conc (ppm)', 'Ethylene conc (ppm)', 'label']])

    # Удаляем лишние столбцы
    df.drop(columns=['Time (seconds)', 'CO conc (ppm)', 'Ethylene conc (ppm)'], inplace=True, errors='ignore')
    print(f"- Используемые столбцы: {df.columns.tolist()}")

    return df
