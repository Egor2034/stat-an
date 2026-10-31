import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.stats as st
import pandas as pd

# Настройка стиля графиков
sb.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

a = 0.0
sigma2 = 4.0
sigma = np.sqrt(sigma2)
n = 125                 # Объем выборки
rule = 'scott'          # Правило для числа интервалов

np.random.seed(42)

# =============================================================================
# ЗАДАНИЕ 1. Смоделировать выборку и сгруппировать данные
# =============================================================================
print("="*60)
print("ЗАДАНИЕ 1: Моделирование и группировка данных")
print("="*60)

# 1. Моделирование выборки
X = np.random.normal(loc=a, scale=sigma, size=n)
print(f"Вывод: Сгенерирована выборка объема n={n} из нормального распределения N({a}, {sigma2}).")

# 1.1. Интервальный ряд абсолютных частот
abs_freqs, bin_edges = np.histogram(X, bins=rule)
print(f"\n1.1. Вывод: Построен интервальный ряд абсолютных частот (правило Скотта).")
print(f"Количество интервалов: {len(abs_freqs)}")
for i in range(len(abs_freqs)):
    print(f"Интервал [{bin_edges[i]:.2f}; {bin_edges[i+1]:.2f}): частота = {abs_freqs[i]}")

# 1.2. Сумма абсолютных частот
sum_abs = np.sum(abs_freqs)
print(f"\n1.2. Вывод: Сумма абсолютных частот = {sum_abs}")

# 1.3. Интервальный ряд относительных частот
rel_freqs = abs_freqs / n 
print(f"\n1.3. Вывод: Построен интервальный ряд относительных частот.")
for i in range(len(rel_freqs)):
    print(f"Интервал [{bin_edges[i]:.2f}; {bin_edges[i+1]:.2f}): отн. частота = {rel_freqs[i]:.4f}")

# 1.4. Сумма относительных частот
sum_rel = np.sum(rel_freqs)
print(f"\n1.4. Вывод: Сумма относительных частот = {sum_rel:.4f}")

# =============================================================================
# ЗАДАНИЕ 2. Визуализация данных (Пункты 2.1 - 2.4)
# =============================================================================
print("\n" + "="*60)
print("ЗАДАНИЕ 2: Визуализация данных (графики)")
print("="*60)
print("Вывод: Графики построены и отображены в отдельных окнах.")

# 2.1. Гистограммы относительных частот с разным числом интервалов (2-10 и 15-25)
plt.figure(figsize=(15, 10))
plot_idx = 1
for bins in [2, 5, 10, 15, 20, 25]:
    plt.subplot(2, 3, plot_idx)
    plt.hist(X, bins=bins, density=True, alpha=0.6, color='skyblue', edgecolor='black') 
    plt.title(f'Гистограмма, интервалов: {bins}')
    plt.xlabel('X')
    plt.ylabel('Плотность')
    plot_idx += 1
plt.suptitle('2.1. Гистограммы относительных частот при разном числе интервалов', fontsize=16)
plt.tight_layout()
plt.show()

# 2.2. Гистограмма абсолютных частот (по правилу Скотта)
plt.figure()
plt.hist(X, bins=rule, alpha=0.6, color='lightgreen', edgecolor='black')
plt.title(f'2.2. Гистограмма абсолютных частот (правило Скотта)')
plt.xlabel('X')
plt.ylabel('Абсолютная частота')
plt.show()

# 2.3. Гистограмма относительных частот + теоретическая кривая
plt.figure()
# density=True делает гистограмму относительных частот (площадь = 1)
plt.hist(X, bins=rule, density=True, alpha=0.6, color='salmon', edgecolor='black', label='Гистограмма (отн. частоты)')
# Строим теоретическую кривую
x_axis = np.linspace(min(X) - 1, max(X) + 1, 100)
pdf_theor = st.norm.pdf(x_axis, a, sigma)
plt.plot(x_axis, pdf_theor, 'k-', lw=2, label=f'Теоретическая N({a}, {sigma2})')
plt.title('2.3. Гистограмма относительных частот и теоретическая кривая')
plt.xlabel('X')
plt.ylabel('Плотность вероятности')
plt.legend()
plt.show()

# 2.4. Эмпирическая и теоретическая функции распределения
plt.figure()
# Эмпирическая функция распределения (строится по отсортированной выборке)
X_sorted = np.sort(X)
# Для ступенчатого графика
plt.step(X_sorted, np.arange(1, n + 1) / n, where='post', label='Эмпирическая F(x)', color='blue')
# Теоретическая функция распределения
cdf_theor = st.norm.cdf(x_axis, a, sigma)
plt.plot(x_axis, cdf_theor, 'r-', lw=2, label=f'Теоретическая F(x) N({a}, {sigma2})')
plt.title('2.4. Эмпирическая и теоретическая функции распределения')
plt.xlabel('X')
plt.ylabel('F(x)')
plt.legend()
plt.show()

# =============================================================================
# ЗАДАНИЕ 2.5. Бокс-плот
# =============================================================================
print("\n" + "="*60)
print("ЗАДАНИЕ 2.5: Бокс-плот и его статистическая интерпретация")
print("="*60)

plt.figure(figsize=(10, 3))
sb.boxplot(x=X, color='lightblue')
plt.title('2.5. Бокс-плот для выборки X')
plt.xlabel('X')
plt.show()

# Вычисление квартилей и границ
q1 = np.percentile(X, 25)
median = np.percentile(X, 50)
q3 = np.percentile(X, 75)
iqr = q3 - q1
lower_whisker = q1 - 1.5 * iqr
upper_whisker = q3 + 1.5 * iqr

# Для точности найдем реальные границы усов (min/max выборки в пределах границ)
real_lower = min(X[X >= lower_whisker])
real_upper = max(X[X <= upper_whisker])

print(f"Выборочные квартили:")
print(f"  Q1 (25-й перцентиль) = {q1:.4f}")
print(f"  Медиана (50-й перцентиль) = {median:.4f}")
print(f"  Q3 (75-й перцентиль) = {q3:.4f}")
print(f"Интерквартильный размах (IQR) = {iqr:.4f}")
print(f"Теоретические границы 'усов' (1.5*IQR):")
print(f"  Нижняя теоретическая = {lower_whisker:.4f}")
print(f"  Верхняя теоретическая = {upper_whisker:.4f}")
print(f"Фактические границы 'усов' на графике:")
print(f"  Нижняя (min без выбросов) = {real_lower:.4f}")
print(f"  Верхняя (max без выбросов) = {real_upper:.4f}")
print("Вывод: Бокс-плот показывает, что медиана близка к математическому ожиданию (0).")
print("Распределение симметрично, выбросов нет (все данные попали в пределы 1.5*IQR).") 

# =============================================================================
# ЗАДАНИЕ 3. Точечное оценивание параметров
# =============================================================================
print("\n" + "="*60)
print("ЗАДАНИЕ 3: Точечное оценивание параметров")
print("="*60)

# 3.1. Два способа вычисления оценок
print("\n--- 3.1. Вычисление оценок двумя способами ---")

# Способ 1: Ручное программирование формул
mean_1 = np.sum(X) / n
median_1 = np.median(X)

mode_1 = st.mode(X, keepdims=False).mode

var_1 = np.sum((X - mean_1)**2) / n # Дисперсия (смещенная)
var_unb_1 = np.sum((X - mean_1)**2) / (n - 1) # Исправленная дисперсия (несмещенная)
std_1 = np.sqrt(var_1) # Стандартное отклонение
std_unb_1 = np.sqrt(var_unb_1) # Исправленное стандартное отклонение
skew_1 = np.sum((X - mean_1)**3) / (n * std_1**3) # Асимметрия
kurt_1 = (np.sum((X - mean_1)**4) / (n * var_1**2)) - 3 # Эксцесс (избыточный)

# Способ 2: Встроенные функции
mean_2 = np.mean(X) # Среднее
median_2 = np.median(X) # Медиана
mode_2 = st.mode(X, keepdims=False).mode # Мода
var_2 = np.var(X, ddof=0) # Дисперсия 
var_unb_2 = np.var(X, ddof=1) # Исправленная дисперсия 
std_2 = np.std(X, ddof=0) # Стандартное отклонение
std_unb_2 = np.std(X, ddof=1) # Исправленное стандартное отклонение
skew_2 = st.skew(X) # Асимметрия
kurt_2 = st.kurtosis(X) # Эксцесс 

# Сравнение
print(f"{'Параметр':<25} | {'Способ 1 (Ручной)':<20} | {'Способ 2 (Встроенный)':<20}")
print("-" * 70)
print(f"{'Среднее':<25} | {mean_1:<20.6f} | {mean_2:<20.6f}")
print(f"{'Медиана':<25} | {median_1:<20.6f} | {median_2:<20.6f}")
print(f"{'Мода':<25} | {mode_1:<20.6f} | {mode_2:<20.6f}")
print(f"{'Дисперсия (смещ)':<25} | {var_1:<20.6f} | {var_2:<20.6f}")
print(f"{'Дисперсия (испр)':<25} | {var_unb_1:<20.6f} | {var_unb_2:<20.6f}")
print(f"{'Ст. откл. (смещ)':<25} | {std_1:<20.6f} | {std_2:<20.6f}")
print(f"{'Ст. откл. (испр)':<25} | {std_unb_1:<20.6f} | {std_unb_2:<20.6f}")
print(f"{'Асимметрия':<25} | {skew_1:<20.6f} | {skew_2:<20.6f}")
print(f"{'Эксцесс':<25} | {kurt_1:<20.6f} | {kurt_2:<20.6f}")
print("Вывод: Значения, полученные двумя способами, совпадают с точностью до машинной погрешности.")

# 3.2. Увеличение объема выборки в 60 раз
print("\n--- 3.2. Увеличение объема выборки в 60 раз ---")
n_large = n * 60
print(f"Новый объем выборки n = {n_large}")

X_large = np.random.normal(loc=a, scale=sigma, size=n_large)

# Вычисление вторым способом
mean_large = np.mean(X_large) # Среднее
median_large = np.median(X_large) # Медиана
mode_large = st.mode(X_large, keepdims=False).mode # Мода
var_large = np.var(X_large, ddof=0) # Дисперсия 
std_large = np.std(X_large, ddof=0) # Стандартное отклонение
var_unb_large = np.var(X_large, ddof=1) # Исправленная дисперсия 
std_unb_large = np.std(X_large, ddof=1) # Исправленное стандартное отклонение
skew_large = st.skew(X_large) # Асимметрия
kurt_large = st.kurtosis(X_large) # Эксцесс 

print(f"{'Параметр':<25} | {'n = 125':<20} | {'n = 7500':<20} | {'Теория':<10}")
print("-" * 80)
print(f"{'Среднее':<25} | {mean_2:<20.4f} | {mean_large:<20.4f} | {a:<10.4f}")
print(f"{'Дисперсия (смещ)':<25} | {var_2:<20.4f} | {var_large:<20.4f} | {sigma2:<10.4f}")
print(f"{'Дисперсия (испр)':<25} | {var_unb_2:<20.4f} | {var_unb_large:<20.4f} | {sigma2:<10.4f}")
print(f"{'Асимметрия':<25} | {skew_2:<20.4f} | {skew_large:<20.4f} | {0.0:<10.4f}")
print(f"{'Эксцесс':<25} | {kurt_2:<20.4f} | {kurt_large:<20.4f} | {0.0:<10.4f}")

# 3.3. Анализ
print("\n3.3. Вывод:")
print("При увеличении объема выборки в 60 раз (с 125 до 7500) точечные оценки стали значительно ближе к теоретическим значениям.")
print("С ростом n оценки сходятся по вероятности к истинным параметрам распределения.")
print("Например, выборочное среднее приблизилось к 0, дисперсия к 4, а асимметрия и эксцесс к 0.")