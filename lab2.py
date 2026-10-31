import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.stats as st

# Настройка стиля графиков
sb.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# =============================================================================
# ПАРАМЕТРЫ ВАРИАНТА 8
# =============================================================================
a = 0.0
sigma2 = 4.0
sigma = np.sqrt(sigma2)
n = 10
N = 450
gamma = 0.945

# Фиксируем seed для воспроизводимости
np.random.seed(8)

# ПУНКТ 1: N выборок по n наблюдений. Вычисление выборочных средних X_bar
X_bar = np.empty(N)
for i in range(N):
    X_i = np.random.normal(a, sigma, n)
    X_bar[i] = np.mean(X_i)

print("\nПУНКТ 1: Смоделированы N выборок, вычислены выборочные средние.")
print(f"Первые 5 значений X_bar: {X_bar[:5]}")

# ПУНКТ 2: Гистограмма X_bar + теоретическая кривая N(a, sigma^2/n)
mean_theor = a
std_theor = sigma / np.sqrt(n) # = 2/sqrt(10) ≈ 0.632

plt.figure()
plt.hist(X_bar, bins='sturges', density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Гистограмма X_bar')
x_axis = np.linspace(min(X_bar), max(X_bar), 200)
pdf_Xbar = st.norm.pdf(x_axis, mean_theor, std_theor)
plt.plot(x_axis, pdf_Xbar, 'r-', lw=2, label=f'Теоретическая N({a}, {sigma2/n:.2f})')
plt.title('ПУНКТ 2: Гистограмма и теоретическая кривая для X_bar')
plt.xlabel('X_bar')
plt.ylabel('Плотность')
plt.legend()
plt.show()

# ПУНКТ 3: Бокс-плот X_bar
plt.figure(figsize=(8, 3))
sb.boxplot(x=X_bar, color='skyblue')
plt.title('ПУНКТ 3: Бокс-плот X_bar')
plt.xlabel('X_bar')
plt.show()

# ПУНКТ 4: Точечные оценки параметров X_bar
mean_Xbar = np.mean(X_bar)           # Выборочное среднее
var_Xbar = np.var(X_bar, ddof=1)     # Выборочная исправленная дисперсия (несмещенная, делим на N-1)
std_Xbar = np.std(X_bar, ddof=1)     # Выборочное исправленное стандартное отклонение
median_Xbar = np.median(X_bar)       # Выборочная медиана (50-й перцентиль)
skew_Xbar = st.skew(X_bar)           # Выборочный коэффициент асимметрии
kurt_Xbar = st.kurtosis(X_bar)       # Выборочный коэффициент эксцесса (избыточный, минус 3)

print("\nПУНКТ 4: Точечные оценки параметров для X_bar:")
print(f"Выборочное среднее: {mean_Xbar:.4f} (теория a = {a})")
print(f"Выборочная дисперсия: {var_Xbar:.4f} (теория sigma^2/n = {sigma2/n:.4f})")
print(f"Станд. отклонение: {std_Xbar:.4f} (теория σ/√n = {std_theor:.4f})")
print(f"Выборочная медиана: {median_Xbar:.4f}")
print(f"Выборочная асимметрия: {skew_Xbar:.4f}")
print(f"Выборочный эксцесс: {kurt_Xbar:.4f}")

# ПУНКТ 5: Доля X_bar > a
pct_gt_a = np.mean(X_bar > a) * 100
print(f"\nПУНКТ 5: Доля значений X_bar > a = {pct_gt_a:.2f}%")

# ПУНКТ 6: Выборочные дисперсии D_v
D_v = np.empty(N)
for i in range(N):
    X_i = np.random.normal(a, sigma, n)
    D_v[i] = np.var(X_i, ddof=0) # смещенная дисперсия (делим на n)

print("\nПУНКТ 6: Сформирована выборка из N значений D_v")
print(f"Первые 5 значений D_v: {D_v[:5]}")

# ПУНКТ 7: Доля D_v < sigma^2
pct_dv_lt = np.mean(D_v < sigma2) * 100
print(f"\nПУНКТ 7: Доля значений D_v < sigma^2 = {pct_dv_lt:.2f}%")

# ПУНКТ 8: Y = n * D_v / sigma^2
df = n - 1
Y = n * D_v / sigma2
print("\nПУНКТ 8: Рассчитана выборка значений Y = n*D_v/sigma^2")
print(f"Первые 5 значений Y: {Y[:5]}")
print(f"Число степеней свободы df = {df}")

# ПУНКТ 9: Гистограмма Y + теоретическая кривая хи-квадрат
plt.figure()
plt.hist(Y, bins='sturges', density=True, alpha=0.6, color='lightgreen', edgecolor='black', label='Гистограмма Y')
y_axis = np.linspace(min(Y), max(Y), 200)
pdf_Y = st.chi2.pdf(y_axis, df=df)
plt.plot(y_axis, pdf_Y, 'r-', lw=2, label=f'Теоретическая χ²({df})')
plt.title('ПУНКТ 9: Гистограмма и теоретическая кривая для Y')
plt.xlabel('Y')
plt.ylabel('Плотность')
plt.legend()
plt.show()

# ПУНКТ 10: Бокс-плот Y
plt.figure(figsize=(8, 3))
sb.boxplot(x=Y, color='lightgreen')
plt.title('ПУНКТ 10: Бокс-плот Y')
plt.xlabel('Y')
plt.show()

# ПУНКТ 11: Теоретические параметры Y
theo_mean_Y = df                     # Мат. ожидание хи-квадрат = числу степеней свободы
theo_var_Y = 2 * df                  # Дисперсия хи-квадрат = 2 * число степеней свободы
theo_median_Y = st.chi2.ppf(0.5, df) # Медиана ищется через функцию ppf
theo_skew_Y = np.sqrt(8 / df)        # Теоретическая асимметрия хи-квадрат
theo_kurt_Y = 12 / df                # Теоретический эксцесс хи-квадрат

print("\nПУНКТ 11: Теоретические значения параметров для Y ~ χ²(n-1):")
print(f"Мат. ожидание: {theo_mean_Y:.4f}")
print(f"Дисперсия: {theo_var_Y:.4f}")
print(f"Медиана: {theo_median_Y:.4f}")
print(f"Асимметрия: {theo_skew_Y:.4f}")
print(f"Эксцесс: {theo_kurt_Y:.4f}")

# ПУНКТ 12: Выборочные оценки Y
mean_Y = np.mean(Y)
var_Y = np.var(Y, ddof=1)
median_Y = np.median(Y)
skew_Y = st.skew(Y)
kurt_Y = st.kurtosis(Y)

print("\nПУНКТ 12: Выборочные оценки параметров для Y:")
print(f"Выборочное среднее: {mean_Y:.4f}")
print(f"Выборочная дисперсия: {var_Y:.4f}")
print(f"Выборочная медиана: {median_Y:.4f}")
print(f"Выборочная асимметрия: {skew_Y:.4f}")
print(f"Выборочный эксцесс: {kurt_Y:.4f}")

# ПУНКТ 13: ДИ для a при известной 
X13 = np.random.normal(a, sigma, n)
x13_mean = np.mean(X13)
alpha = 1 - gamma
z_q = st.norm.ppf(1 - alpha / 2)
half = z_q * sigma / np.sqrt(n)
ci_known = (x13_mean - half, x13_mean + half)
ci_known_scipy = st.norm.interval(gamma, loc=x13_mean, scale=sigma / np.sqrt(n))

print("\nПУНКТ 13: ДИ для a при известной sigma^2:")
print(f"Выборочное среднее: {x13_mean:.4f}")
print(f"ДИ (ppf): ({ci_known[0]:.4f}; {ci_known[1]:.4f})")
print(f"ДИ (interval): ({ci_known_scipy[0]:.4f}; {ci_known_scipy[1]:.4f})")