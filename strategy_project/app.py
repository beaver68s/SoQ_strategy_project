import streamlit as st
import pandas as pd
import plotly.express as px
import os

PATH = ""
st.set_page_config(layout="wide")

st.title("Анализ Hurst и корреляций — Домашняя работа по предмету")

# Вводная информация
st.markdown("""
    Это приложение является домашней работой по предмету "Анализ временных рядов".
    В нем представлены различные визуализации данных, связанные с анализом Hurst-экспоненты, 
    вариационной доли (Variance Ratio) и корреляций между различными активами.
    
    Приложение позволяет:
    - Исследовать значения Hurst для разных источников данных,
    - Анализировать доверительные интервалы для Hurst-экспоненты,
    - Оценивать Variance Ratio для различных лагов (q),
    - Исследовать 3D зависимость корреляции от параметров LookBack и Hold.
    
    Вы можете взаимодействовать с графиками и фильтровать данные для анализа.
""")

# st.subheader(f"{os.getcwd()}")

# --- Загрузка и визуализация данных Hurst Value ---
st.subheader("График Hurst значений для разных источников данных")
st.markdown("""
    В этом отчете представлены различные визуализации данных, включая:
    - Графики Hurst значений,
    - Графики с доверительными интервалами,
    - Линейные графики Variance Ratio,
    - 3D визуализация зависимости корреляции от параметров LookBack и Hold.
    Вы можете взаимодействовать с графиками и фильтровать данные для анализа.
""")

# --- Загрузка и визуализация данных Hurst Value ---
st.subheader("График Hurst значений для разных источников данных")

# Путь к файлу с данными Hurst
data_path = os.path.join(PATH, "data", "results", "hurst_data.csv")

# Загрузка данных Hurst из CSV-файла
hurst_df = pd.read_csv(data_path)

# Визуализация графика для Hurst значений
fig = px.bar(
    hurst_df.sort_values(by='hurst_value'),  # Сортируем по значениям Hurst
    x='hurst_value',  # Значения Hurst на оси X
    y='data',  # Источник данных на оси Y
    title='Hurst Values by Data Source',  # Заголовок графика
    labels={'hurst_value': 'Hurst Value', 'data': 'Data'},  # Подписи осей
    range_x=[0.4, 0.6]  # Ограничение по оси X от 0.4 до 0.6
)

# Добавляем вертикальную линию на значении 0.5 (Random Walk)
fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Random Walk")

left, center, right = st.columns([1, 2, 1])
with center:
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
    Этот график отображает значения Hurst для разных источников данных. 
    Вы можете увидеть, как значения Hurst распределяются в диапазоне от 0.4 до 0.6.
    Красная вертикальная линия на уровне 0.5 показывает значение для случайного блуждания (Random Walk).
""")

# --- Второй график: Hurst значений с доверительными интервалами ---
st.subheader("Hurst Exponent с 95% доверительными интервалами")

# Путь к файлу с данными для доверительных интервалов Hurst
hurst_ci_path = os.path.join(PATH, "data", "results", "hurst_ci_data.csv")
hurst_df_ci = pd.read_csv(hurst_ci_path)

# Расчет ошибок для доверительных интервалов
error_x = hurst_df_ci['ci_high'] - hurst_df_ci['hurst_value']
error_x_minus = hurst_df_ci['hurst_value'] - hurst_df_ci['ci_low']

col1, col2 = st.columns([3, 4])  # Первая колонка для графика, вторая — для кода

# Вставляем график в первую колонку
with col1:
    fig2 = px.bar(
        hurst_df_ci.sort_values(by='hurst_value'),
        x='hurst_value',
        y='data',
        error_x=error_x,  # Ошибка по положительному пределу
        error_x_minus=error_x_minus,  # Ошибка по отрицательному пределу
        title='Hurst Exponent with 95% Confidence Intervals',  # Заголовок графика
        labels={'hurst_value': 'Hurst Value', 'data': 'Data'},  # Подписи осей
        range_x=[0.4, 0.6]  # Ограничение по оси X от 0.4 до 0.6
    )
    fig2.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Random Walk")

    st.plotly_chart(fig2)

# Вставляем текст с кодом во вторую колонку
with col2:
    st.markdown('<span style="font-size: 24px;">Код для Bootstrap-метода Hurst Exponent:</span>', unsafe_allow_html=True)

    code = '''def bootstrap_hurst(series, n_iter=100, sample_frac=0.6, n_jobs=-1):
    series = np.asarray(series)
    n = len(series)
    sample_size = int(n * sample_frac)

    def one_bootstrap():
        idx = np.random.choice(n, size=sample_size, replace=False)
        sample = series[np.sort(idx)]
        H, _, _ = compute_Hc(sample, kind='price', simplified=False)
        return H

    hurst_vals = Parallel(n_jobs=n_jobs)(
        delayed(one_bootstrap)() for _ in tqdm(range(n_iter), desc='Bootstaping...')
    )

    return np.percentile(hurst_vals, [2.5, 97.5]), np.mean(hurst_vals)'''
    st.code(code, language='python', )

st.markdown("""
    Этот график отображает значения Hurst с доверительными интервалами на 95%.
    Ошибки показывают диапазон значений, в котором с вероятностью 95% находится истинное значение Hurst.
    Также добавлена красная линия на уровне 0.5, показывающая значение для случайного блуждания.
""")

# --- Третий график: Variance Ratio vs Lag q ---
st.subheader("Variance Ratio (VR) vs Lag q")

# Путь к файлу с данными для Variance Ratio
vr_df_path = os.path.join(PATH, "data", "results", "anrew_lo_test.parquet")
vr_df = pd.read_parquet(vr_df_path)

# Визуализация линейного графика Variance Ratio
fig3 = px.line(
    vr_df,
    x='q',  # Параметр q (лаг)
    y='VR',  # Variance Ratio
    color='asset',  # Разделение по типу актива
    markers=True,  # Отображение маркеров на линии
    labels={'q': 'Lag q', 'VR': 'Variance Ratio', 'asset': 'Asset'},  # Подписи осей
    title='Variance Ratio vs Lag q'  # Заголовок графика
)

# Добавляем горизонтальную линию на уровне y=1.0 (Random Walk)
fig3.add_hline(
    y=1.0,
    line_dash="dash",
    line_color="gray",
    annotation_text="Random Walk",
    annotation_position="bottom right"
)

# Настройка отображения сетки и отступов
fig3.update_layout(
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True),
    legend=dict(title='Asset'),
    margin=dict(l=40, r=40, t=60, b=40)
)

# Отображаем график в Streamlit
st.plotly_chart(fig3)

st.markdown("""
    Этот график показывает Variance Ratio для различных значений параметра q (лаг).
    Линия на уровне 1.0 (Random Walk) помогает понять, как Variance Ratio сравнивается с ожидаемым значением для случайного блуждания.
""")

# --- Четвёртый график: 3D Scatter Correlation vs LookBack/Hold ---
st.subheader("3D визуализация зависимости корреляции от LookBack и Hold")

# Путь к файлу с данными для корреляции
returns_table_path = os.path.join(PATH, "data", "results", "correlation_table.parquet")
returns_table = pd.read_parquet(returns_table_path)

# Переименование столбцов для удобства
returns_table = returns_table.rename(columns={
    'Abs(corr)': 'Correlation_abs',  # Абсолютное значение корреляции
    'data': 'Asset'  # Котировка актива
})

# Уникальные значения активов (котировок)
available_assets = returns_table['Asset'].unique()

# Выбор котировки через Streamlit
selected_asset = st.selectbox("Выберите котировку:", available_assets)

# Фильтрация данных по выбранному активу
filtered_df = returns_table[returns_table['Asset'] == selected_asset]

# Построение 3D-графика с цветовой шкалой от зеленого до оранжевого
fig_3d = px.scatter_3d(
    filtered_df,
    x="LookBack",  # Параметр LookBack (время задержки)
    y="Hold",  # Параметр Hold (удержание)
    z="Correlation_abs",  # Абсолютная корреляция
    color="Abs(corr / p_value (H1: corr!=0))",  # Цвет на основе значения корреляции
    size="Correlation_abs",  # Размер точек на основе абсолютной корреляции
    title=f"3D визуализация зависимости корреляции от LookBack и Hold для {selected_asset}",
    color_continuous_scale=["green", "orange"],  # Цветовая шкала от зеленого до оранжевого
    height= 600
)

# Настройка осей в 3D-графике
fig_3d.update_layout(scene=dict(
    xaxis_title='LookBack (мин)',
    yaxis_title='Hold (мин)',
    zaxis_title='Correlation'
))

left, center, right = st.columns([1, 2, 1])
with center:
    st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("""
    Этот 3D-график отображает зависимость корреляции от параметров LookBack и Hold для выбранной котировки.
    Цвет точек на графике показывает степень корреляции, а их размер зависит от величины корреляции.
    Выберите актив (котировку) из выпадающего списка для анализа данных.
""")


st.subheader("Бэктест простейшей intra-day стратегии")

col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
with col2:
    st.image(PATH+"data/pictures/south_park1.png", caption="Офис Binance", use_column_width=True)
with col3:
    st.image(PATH+"data/pictures/south_park2.png", use_column_width=True)

backtesting_data = pd.read_parquet(PATH + 'data/results/backtesting0.parquet')
col1, col2, col4 = st.columns([1, 2, 1])
with col2:
    fig_3d = px.scatter_3d(
        backtesting_data,
        x="X",
        y="slippage_usd",
        z="total_return",
        color="sharpe",
        title="3D визуализация бектестинга",
        height=600,
        hover_data=['drawdown'],
        color_continuous_scale=["green", "orange"],
    )

    st.plotly_chart(fig_3d, use_container_width=True)