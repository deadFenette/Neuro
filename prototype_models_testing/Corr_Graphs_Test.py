import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# Оптимизированные параметры модели нейрона Ижикевича
default_params = {
    'a': 0.02,
    'b': 0.2,
    'c': -65,
    'd': 8
}

# Векторизованная функция внешнего тока
def external_current(t: np.ndarray) -> np.ndarray:
    I_ext = np.zeros_like(t)
    I_ext[(100 < t) & (t < 200)] = 10
    I_ext[(300 < t) & (t < 400)] = 15
    return I_ext

# Векторизованная модель Ижикевича с коррекцией Жукова
def izhikevich_model_zk(t: np.ndarray, y: np.ndarray, params: Dict[str, float], dt: float) -> np.ndarray:
    v, u = y
    I_ext = external_current(t)
    dvdt = 0.04 * v ** 2 + 5 * v + 140 - u + I_ext
    dudt = params['a'] * (params['b'] * v - u)

    # Коррекция Жукова
    v_next = v + dt * dvdt
    u_next = u + dt * dudt
    dvdt_next = 0.04 * v_next ** 2 + 5 * v_next + 140 - u_next + I_ext
    dudt_next = params['a'] * (params['b'] * v_next - u_next)

    dvdt_corr = (dvdt + dvdt_next) / 2
    dudt_corr = (dudt + dudt_next) / 2

    return np.array([dvdt_corr, dudt_corr])

# Векторизованная модель Ижикевича без коррекции
def izhikevich_model(t: np.ndarray, y: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    v, u = y
    I_ext = external_current(t)
    dvdt = 0.04 * v ** 2 + 5 * v + 140 - u + I_ext
    dudt = params['a'] * (params['b'] * v - u)
    return np.array([dvdt, dudt])

# Функция для симуляции модели
def simulate_model(params: Dict[str, float], t_max: float, dt: float, v_threshold: float, use_zk: bool = False) -> \
Tuple[np.ndarray, np.ndarray]:
    y0 = np.array([-65, -13])
    all_t = []
    all_v = []
    all_u = []
    t_start = 0

    while t_start < t_max:
        t_end = min(t_start + dt * 1000, t_max)  # t_end не должно превышать t_max
        t_values = np.arange(t_start, t_end, dt)

        if use_zk:
            sol = solve_ivp(
                lambda t, y: izhikevich_model_zk(t, y, params, dt),
                (t_start, t_end),
                y0,
                method='RK45',  # Использование адаптивного шага времени
                t_eval=t_values,
                vectorized=True
            )
        else:
            sol = solve_ivp(
                lambda t, y: izhikevich_model(t, y, params),
                (t_start, t_end),
                y0,
                method='RK45',  # Использование адаптивного шага времени
                t_eval=t_values,
                vectorized=True
            )

        all_t.extend(sol.t)
        all_v.extend(sol.y[0])
        all_u.extend(sol.y[1])

        t_start = t_end
        y0 = sol.y[:, -1]  # продолжаем с последнего значения

    return np.array(all_t), np.array([all_v, all_u])

# Функция для сравнения моделей с коррекцией Жукова и без нее
def compare_models(params: Dict[str, float], t_max: float, dt: float, v_threshold: float):
    t_zk, sol_zk = simulate_model(params, t_max, dt, v_threshold, use_zk=True)
    t_no_zk, sol_no_zk = simulate_model(params, t_max, dt, v_threshold, use_zk=False)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t_zk, sol_zk[0], label='Мембранный потенциал с коррекцией Жукова', color='blue')
    plt.plot(t_no_zk, sol_no_zk[0], label='Мембранный потенциал без коррекции Жукова', linestyle='--', color='red')
    plt.axhline(y=default_params['c'], color='gray', linestyle='--', label='Порог')
    plt.xlabel('Время (мс)')
    plt.ylabel('Мембранный потенциал (мВ)')
    plt.title('Сравнение модели нейрона Ижикевича')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t_zk, sol_zk[1], label='Переменная восстановления с коррекцией Жукова', color='green')
    plt.plot(t_no_zk, sol_no_zk[1], label='Переменная восстановления без коррекции Жукова', linestyle='--', color='orange')
    plt.xlabel('Время (мс)')
    plt.ylabel('Переменная восстановления (u)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Задаем параметры для сравнения
params = {
    'a': 0.02,
    'b': 0.2,
    'c': -65,
    'd': 8
}

# Задаем время симуляции и шаг
t_max = 500  # мс
dt = 0.1  # мс

# Запускаем сравнение моделей
compare_models(params, t_max, dt, params['c'])