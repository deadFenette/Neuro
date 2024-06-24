import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# Функция модели Ижикевича
def izhikevich_model(v, u, I, a, b, c, d):
    dvdt = 0.04 * v ** 2 + 5 * v + 140 - u + I
    dudt = a * (b * v - u)
    return dvdt, dudt


# Метод PK2 для решения ОДУ
def pk2_step(v, u, I, a, b, c, d, dt):
    k1_v, k1_u = izhikevich_model(v, u, I, a, b, c, d)
    v1 = v + k1_v * dt / 2
    u1 = u + k1_u * dt / 2
    k2_v, k2_u = izhikevich_model(v1, u1, I, a, b, c, d)
    v_new = v + k2_v * dt
    u_new = u + k2_u * dt
    return v_new, u_new


# Коррекция Жукова
def zukov_correction(v, u, v_prev, u_prev, dt):
    return v + (v - v_prev) * dt, u + (u - u_prev) * dt


# Модель с и без коррекции Жукова
def simulate_izhikevich(I_ext_intervals, a, b, c, d, v_init=-65, u_init=0, dt=0.1, t_max=100):
    t = np.arange(0, t_max, dt)
    v_no_zk = np.zeros_like(t)
    u_no_zk = np.zeros_like(t)
    v_zk = np.zeros_like(t)
    u_zk = np.zeros_like(t)

    v = v_init
    u = u_init
    v_prev = v
    u_prev = u

    I_ext = np.zeros_like(t)

    for interval in I_ext_intervals:
        t_start, t_end, I_value = interval
        idx_start = int(t_start / dt)
        idx_end = int(t_end / dt)
        I_ext[idx_start:idx_end] = I_value

    for i in range(len(t)):
        I = I_ext[i]

        v_no_zk[i] = v
        u_no_zk[i] = u
        v, u = pk2_step(v, u, I, a, b, c, d, dt)

        if v >= 30:  # Порог спайка
            v = c
            u += d

        v_zk[i] = v
        u_zk[i] = u
        v, u = zukov_correction(v, u, v_prev, u_prev, dt)
        v_prev, u_prev = v_zk[i], u_zk[i]

    return t, v_no_zk, u_no_zk, v_zk, u_zk


# Функция для ввода интервала I_ext с клавиатуры
def input_I_ext_intervals(dt):
    print("Введите интервалы времени и значения I_ext для каждого интервала:")
    I_ext_intervals = []
    while True:
        t_start_str = input("Начальное время интервала (для завершения введите пустое значение): ")
        if t_start_str == '':
            break
        t_start = float(t_start_str)
        t_end = float(input("Конечное время интервала: "))
        I_value = float(input(f"Значение I_ext для интервала [{t_start}, {t_end}]: "))
        I_ext_intervals.append((t_start, t_end, I_value))
    return I_ext_intervals


# Функция для вычисления метрик точност
def compute_metrics(sol_no_zk, sol_zk):
    mask = sol_zk != 0  # Исключаем деление на ноль
    mape = np.mean(np.abs((sol_no_zk[mask] - sol_zk[mask]) / sol_zk[mask])) * 100 if np.any(mask) else np.nan
    mse = np.mean((sol_no_zk - sol_zk) ** 2)
    mae = np.mean(np.abs(sol_no_zk - sol_zk))
    rmse = np.sqrt(mse)
    r2 = 1 - (mse / np.var(sol_zk))
    return mse, mae, rmse, r2, mape


# Функция для вычисления скользящего среднего
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


# Основная функция для выполнения модели
def main():
    a = float(input("Введите параметр a: "))
    b = float(input("Введите параметр b: "))
    c = float(input("Введите параметр c: "))
    d = float(input("Введите параметр d: "))
    dt = float(input("Введите шаг времени dt: "))
    t_max = float(input("Введите максимальное время t_max: "))

    # Ввод интервалов I_ext
    I_ext_intervals = input_I_ext_intervals(dt)

    v_init = -65
    u_init = 0

    t, v_no_zk, u_no_zk, v_zk, u_zk = simulate_izhikevich(I_ext_intervals, a, b, c, d, v_init, u_init, dt, t_max)

    # Интерполяция для вычисления метрик
    t_interp = np.linspace(0, t_max, len(t))
    v_no_zk_interp = interp1d(t, v_no_zk, fill_value="extrapolate")(t_interp)
    v_zk_interp = interp1d(t, v_zk, fill_value="extrapolate")(t_interp)

    # Вычисление метрик точности для обеих моделей
    mse_no_zk, mae_no_zk, rmse_no_zk, r2_no_zk, mape_no_zk = compute_metrics(v_no_zk_interp, v_no_zk_interp)
    mse_zk, mae_zk, rmse_zk, r2_zk, mape_zk = compute_metrics(v_no_zk_interp, v_zk_interp)

    # Вывод метрик
    print("Метрики без коррекции Жукова:")
    print(f"MSE: {mse_no_zk}")
    print(f"MAE: {mae_no_zk}")
    print(f"RMSE: {rmse_no_zk}")
    print(f"R2: {r2_no_zk}")
    print(f"MAPE: {mape_no_zk}%\n")

    print("Метрики с коррекцией Жукова:")
    print(f"MSE: {mse_zk}")
    print(f"MAE: {mae_zk}")
    print(f"RMSE: {rmse_zk}")
    print(f"R2: {r2_zk}")
    print(f"MAPE: {mape_zk}%\n")

    # Построение графиков
    plt.figure(figsize=(12, 6))
    plt.plot(t, v_no_zk, label='Без коррекции Жукова')
    plt.plot(t, v_zk, label='С коррекцией Жукова')
    plt.xlabel('Время')
    plt.ylabel('Мембранный потенциал v')
    plt.legend()
    plt.title('Модель нейрона Ижикевича')
    plt.show()

    # Скользящее среднее для v, u и частоты спайков
    window_size = int(100 / dt)  # Скользящее окно для 100 мс
    v_no_zk_ma = moving_average(v_no_zk, window_size)
    v_zk_ma = moving_average(v_zk, window_size)
    u_no_zk_ma = moving_average(u_no_zk, window_size)
    u_zk_ma = moving_average(u_zk, window_size)

    # Частота спайков
    spikes_no_zk = np.where(v_no_zk >= 30, 1, 0)
    spikes_zk = np.where(v_zk >= 30, 1, 0)
    spike_freq_no_zk = moving_average(spikes_no_zk, window_size) / (window_size * dt) * 1000  # Частота в Гц
    spike_freq_zk = moving_average(spikes_zk, window_size) / (window_size * dt) * 1000  # Частота в Гц

    # График скользящего среднего для v
    plt.figure(figsize=(12, 6))
    plt.plot(t, v_no_zk, label='Без коррекции Жукова')
    plt.plot(t, v_zk, label='С коррекцией Жукова')
    plt.plot(t[:-window_size + 1], v_no_zk_ma, linestyle='--', label=f'Скользящее среднее ({window_size} точек)')
    plt.plot(t[:-window_size + 1], v_zk_ma, linestyle='--', label=f'Скользящее среднее ({window_size} точек)')
    plt.xlabel('Время')
    plt.ylabel('Мембранный потенциал v')
    plt.legend()
    plt.title('Модель нейрона Ижикевича и скользящее среднее')
    plt.show()

    # График скользящего среднего для u
    plt.figure(figsize=(12, 6))
    plt.plot(t, u_no_zk, label='Без коррекции Жукова')
    plt.plot(t, u_zk, label='С коррекцией Жукова')
    plt.plot(t[:-window_size + 1], u_no_zk_ma, linestyle='--', label=f'Скользящее среднее ({window_size} точек)')
    plt.plot(t[:-window_size + 1], u_zk_ma, linestyle='--', label=f'Скользящее среднее ({window_size} точек)')
    plt.xlabel('Время')
    plt.ylabel('Параметр u')
    plt.legend()
    plt.title('Модель нейрона Ижикевича и скользящее среднее для параметра u')
    plt.show()

    # График частоты спайков
    plt.figure(figsize=(12, 6))
    plt.plot(t[:-window_size + 1], spike_freq_no_zk, label='Без коррекции Жукова')
    plt.plot(t[:-window_size + 1], spike_freq_zk, label='С коррекцией Жукова')
    plt.xlabel('Время')
    plt.ylabel('Частота спайков (Гц)')
    plt.legend()
    plt.title('Частота спайков в модели нейрона Ижикевича')
    plt.show()


if __name__ == "__main__":
    main()