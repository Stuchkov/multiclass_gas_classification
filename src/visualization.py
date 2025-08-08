import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def plot_all_sensors_separately(df, downsample_factor=100, class_names=None):
    """
    Визуализирует все сенсоры и истинные метки временного ряда.
    Делает downsampling для ускорения отрисовки.
    
    Args:
        df (pd.DataFrame): датафрейм с колонками сенсоров и колонкой 'label'.
        downsample_factor (int): шаг сэмплирования (например, 100).
        class_names (list): список имён классов (например, ['Нет газа', 'Ethylene', 'CO', 'Смесь']).
    """
    df_downsampled = df.iloc[::downsample_factor].copy()
    df_downsampled['Время'] = df_downsampled.index  # временная ось по индексу
    
    sensor_features = [col for col in df_downsampled.columns if 'Sensor' in col]
    num_sensors = len(sensor_features)
    
    # Создаём подграфики: первый для меток, остальные для каждого сенсора
    fig = make_subplots(
        rows=num_sensors + 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        subplot_titles=['Истинные метки'] + [f'Показания {s}' for s in sensor_features]
    )
    
    # График меток (верхний)
    fig.add_trace(go.Scatter(
        x=df_downsampled['Время'],
        y=df_downsampled['label'],
        mode='lines',
        line=dict(color='orange', width=1, shape='hv'),
        hovertemplate='Время: %{x}<br>Истинная метка: %{y}<extra></extra>'
    ), row=1, col=1)
    
    # Графики для каждого сенсора
    for i, feature_name in enumerate(sensor_features):
        fig.add_trace(go.Scatter(
            x=df_downsampled['Время'],
            y=df_downsampled[feature_name],
            mode='lines',
            line=dict(color='black', width=1),
            hovertemplate='Время: %{x}<br>Показание: %{y}<extra></extra>'
        ), row=i+2, col=1)
    
    # Настройка отображения осей Y для меток с подписью классов
    if class_names:
        tickvals = list(range(len(class_names)))
        ticktext = class_names
    else:
        tickvals = list(np.unique(df_downsampled['label'].dropna()))
        ticktext = [str(val) for val in tickvals]
    
    fig.update_yaxes(
        title_text='Метка', row=1, col=1,
        tickmode='array',
        tickvals=tickvals,
        ticktext=ticktext
    )
    
    # Подписываем y-оси для сенсоров
    for i, feature_name in enumerate(sensor_features):
        fig.update_yaxes(title_text=feature_name, row=i+2, col=1)
    
    fig.update_layout(
        title_text='Визуализация истинных меток и данных с датчиков',
        height=200 * (num_sensors + 1),
        showlegend=False,
    )
    
    fig.show()
    
def plot_windows(df, sensor_col, windows_info, sampling_rate_hz, time_start_sec, time_end_sec, n_windows_to_plot=80):
    """
    Визуализация окон на одном сенсоре с подписями меток.

    Args:
        df (pd.DataFrame): исходный DataFrame с данными.
        sensor_col (str): колонка сенсора для визуализации.
        windows_info (list of dict): информация об окнах с ключами 'start', 'end', 'label'.
        sampling_rate_hz (int): частота дискретизации данных.
        time_start_sec (float): начало периода по времени для отображения.
        time_end_sec (float): конец периода по времени для отображения.
        n_windows_to_plot (int): количество окон для отображения.
    """
    signal = df[sensor_col].values
    time = np.arange(len(signal)) / sampling_rate_hz

    time_mask = (time >= time_start_sec) & (time <= time_end_sec)
    time_limited = time[time_mask]
    signal_limited = signal[time_mask]

    plt.figure(figsize=(16, 6))

    plt.plot(time_limited, signal_limited, color='gray', linewidth=1.2, label=f'Signal {sensor_col}')

    if len(signal_limited) > 0:
        ymin = np.min(signal_limited)
        ymax = np.max(signal_limited)
        signal_range = ymax - ymin
        base_shift = 0.2 * signal_range
        ylim_lower = ymin - 3 * base_shift
        ylim_upper = ymax + 3 * base_shift
        plt.ylim(ylim_lower, ylim_upper)
    else:
        plt.ylim(0, 1)
        base_shift = 0.2

    group_size = 6

    windows_in_range = []
    for w in windows_info:
        win_start_sec = w['start'] / sampling_rate_hz
        win_end_sec = w['end'] / sampling_rate_hz
        if win_end_sec >= time_start_sec and win_start_sec <= time_end_sec:
            windows_in_range.append(w)

    windows_to_plot = windows_in_range[:n_windows_to_plot]

    for i, win in enumerate(windows_to_plot):
        start_idx_df_co = win['start']
        end_idx_df_co = win['end']

        if end_idx_df_co > len(signal):
            continue

        win_time = time[start_idx_df_co:end_idx_df_co]
        win_signal = signal[start_idx_df_co:end_idx_df_co]

        mask = (win_time >= time_start_sec) & (win_time <= time_end_sec)
        if not any(mask):
            continue

        pos_in_group = i % group_size
        if pos_in_group < 3:
            level = pos_in_group
            shift = base_shift * (level + 1)
        else:
            level = pos_in_group - 3
            shift = -base_shift * (level + 1)

        shifted_signal = win_signal[mask] + shift

        plt.plot(win_time[mask], shifted_signal, color='blue', linewidth=1)

        mid_index = len(win_time[mask]) // 2
        text_x = win_time[mask][mid_index]
        text_y = shifted_signal[mid_index]

        plt.text(text_x, text_y, win['label'],
                 ha='center', va='center',
                 fontsize=10, color='black',
                 bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2'))

    plt.title(f'Signal {sensor_col}, windows with labels')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Sensor data')
    plt.xlim(time_start_sec, time_end_sec)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
