import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# Оптимизированные параметры модели нейрона Ижикевича
default_params = {
    'a': 0.02,
    'b': 0.2,
    'c': -65,
    'd': 8
}


# Конструктор класса модели Ижикевича
class IzhikevichModel:
    def __init__(self, params: Dict[str, float], I_ext: np.ndarray, dt: float):
        self.params = params
        self.I_ext = I_ext
        self.dt = dt
        self.t_array = np.arange(0, len(I_ext) * dt, dt)

    # Модель Ижикевича с коррекцией Жукова
    def model_zk(self, t: float, y: np.ndarray) -> np.ndarray:
        v, u = y
        I_ext_t = np.interp(t, self.t_array, self.I_ext)
        dvdt = 0.04 * v ** 2 + 5 * v + 140 - u + I_ext_t
        dudt = self.params['a'] * (self.params['b'] * v - u)

        v_next = v + self.dt * dvdt
        u_next = u + self.dt * dudt

        dvdt_next = 0.04 * v_next ** 2 + 5 * v_next + 140 - u_next + I_ext_t
        dudt_next = self.params['a'] * (self.params['b'] * v_next - u_next)

        dvdt_corr = (dvdt + dvdt_next) / 2
        dudt_corr = (dudt + dudt_next) / 2

        return np.array([dvdt_corr, dudt_corr])

    # Стандартная модель Ижикевича
    def model(self, t: float, y: np.ndarray) -> np.ndarray:
        v, u = y
        I_ext_t = np.interp(t, self.t_array, self.I_ext)
        dvdt = 0.04 * v ** 2 + 5 * v + 140 - u + I_ext_t
        dudt = self.params['a'] * (self.params['b'] * v - u)
        return np.array([dvdt, dudt])

    def simulate(self, t_max: float, v_threshold: float, use_zk: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        y0 = np.array([-65, -13])
        all_t, all_v, all_u = [], [], []
        t_start = 0
        while t_start < t_max:
            t_end = min(t_start + self.dt * 1000, t_max)

            if use_zk:
                sol = solve_ivp(
                    lambda t, y: self.model_zk(t, y),
                    (t_start, t_end),
                    y0,
                    method='BDF',
                    t_eval=np.linspace(t_start, t_end, int((t_end - t_start) / self.dt) + 1),
                    events=lambda t, y: self.spike_event(t, y, v_threshold),
                    vectorized=True
                )
            else:
                sol = solve_ivp(
                    lambda t, y: self.model(t, y),
                    (t_start, t_end),
                    y0,
                    method='BDF',
                    t_eval=np.linspace(t_start, t_end, int((t_end - t_start) / self.dt) + 1),
                    events=lambda t, y: self.spike_event(t, y, v_threshold),
                    vectorized=True
                )

            all_t.extend(sol.t)
            all_v.extend(sol.y[0])
            all_u.extend(sol.y[1])

            if sol.t_events[0].size > 0:
                spike_times = sol.t_events[0]
                for spike_time in spike_times:
                    idx = np.searchsorted(sol.t, spike_time)
                    if idx < sol.y.shape[1]:
                        y0 = self.reset_condition(sol.y[0, idx], sol.y[1, idx], self.params['c'], self.params['d'])
                        t_start = sol.t[idx] + self.dt
                    else:
                        t_start += self.dt
                        break
            else:
                t_start = t_end
                y0 = sol.y[:, -1]

        return np.array(all_t), np.array([all_v, all_u])

    @staticmethod
    def spike_event(t: float, y: np.ndarray, v_threshold: float) -> float:
        return y[0] - v_threshold

    @staticmethod
    def reset_condition(v: float, u: float, c: float, d: float) -> Tuple[float, float]:
        return c, u + d


def get_external_current(t_max: float, dt: float) -> np.ndarray:
    t = np.arange(0, t_max, dt)
    I_ext = np.zeros_like(t)
    print("Введите интервалы времени и соответствующие значения I_ext.")
    print("Для завершения ввода оставьте интервал времени пустым.")
    while True:
        start_str = input("Начало интервала (в мс): ")
        if not start_str:
            break
        end_str = input("Конец интервала (в мс): ")
        value_str = input("Значение I_ext: ")
        start = float(start_str)
        end = float(end_str)
        value = float(value_str)

        I_ext[(t >= start) & (t < end)] = value

    return I_ext


def calculate_spike_frequency(times: np.ndarray, membrane_potentials: np.ndarray, v_threshold: float, window: float,
                              step: float) -> Tuple[np.ndarray, np.ndarray]:
    spike_indices = np.where(membrane_potentials > v_threshold)[0]
    spike_times = times[spike_indices]
    start_times = np.arange(0, times[-1] - window, step)
    spike_counts = np.array([np.sum((spike_times >= start) & (spike_times < start + window)) for start in start_times])
    spike_rates = spike_counts / (window / 1000)
    return start_times, spike_rates


def compare_models(params: Dict[str, float], t_max: float, dt: float, v_threshold: float, I_ext: np.ndarray):
    model = IzhikevichModel(params, I_ext, dt)
    t_zk, sol_zk = model.simulate(t_max, v_threshold, use_zk=True)
    t_no_zk, sol_no_zk = model.simulate(t_max, v_threshold, use_zk=False)

    # Calculate spike frequencies
    window, step = 100, 10
    start_times_zk, spike_frequency_zk = calculate_spike_frequency(t_zk, sol_zk[0], v_threshold, window, step)
    start_times_no_zk, spike_frequency_no_zk = calculate_spike_frequency(t_no_zk, sol_no_zk[0], v_threshold, window, step)

    # Interpolate solutions for plotting
    t_common = np.linspace(0, t_max, int(t_max / dt))
    sol_zk_interp = np.interp(t_common, t_zk, sol_zk[0])
    sol_no_zk_interp = np.interp(t_common, t_no_zk, sol_no_zk[0])

    # Calculate metrics
    mse_zk = np.mean((sol_zk_interp - sol_no_zk_interp) ** 2)
    mae_zk = np.mean(np.abs(sol_zk_interp - sol_no_zk_interp))
    rmse_zk = np.sqrt(mse_zk)
    r2_zk = 1 - (mse_zk / np.var(sol_no_zk_interp))
    mape_zk = np.mean(np.abs((sol_zk_interp - sol_no_zk_interp) / sol_no_zk_interp)) * 100

    mse_no_zk = np.mean((sol_no_zk_interp - sol_zk_interp) ** 2)
    mae_no_zk = np.mean(np.abs(sol_no_zk_interp - sol_zk_interp))
    rmse_no_zk = np.sqrt(mse_no_zk)
    r2_no_zk = 1 - (mse_no_zk / np.var(sol_zk_interp))
    mape_no_zk = np.mean(np.abs((sol_no_zk_interp - sol_zk_interp) / sol_zk_interp)) * 100

    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(t_zk, sol_zk[0], label='Zhukov correction')
    axs[0, 0].plot(t_no_zk, sol_no_zk[0], label='No Zhukov correction')
    axs[0, 0].set_xlabel('Time (ms)')
    axs[0, 0].set_ylabel('Membrane potential (mV)')
    axs[0, 0].set_title('Membrane potential over time')
    axs[0, 0].legend()

    axs[0, 1].plot(start_times_zk, spike_frequency_zk, label='Zhukov correction')
    axs[0, 1].plot(start_times_no_zk, spike_frequency_no_zk, label='No Zhukov correction')
    axs[0, 1].set_xlabel('Time (ms)')
    axs[0, 1].set_ylabel('Spike frequency (Hz)')
    axs[0, 1].set_title('Spike frequency over time')
    axs[0, 1].legend()

    axs[1, 0].plot(sol_zk_interp, label='Zhukov correction')
    axs[1, 0].plot(sol_no_zk_interp, label='No Zhukov correction')
    axs[1, 0].set_xlabel('Time (ms)')
    axs[1, 0].set_ylabel('Membrane potential (mV)')
    axs[1, 0].set_title('Interpolated membrane potential')
    axs[1, 0].legend()

    axs[1, 1].bar(['Zhukov correction', 'No Zhukov correction'], [mse_zk, mse_no_zk])
    axs[1, 1].set_xlabel('Model')
    axs[1, 1].set_ylabel('MSE')
    axs[1, 1].set_title('Mean squared error')

    plt.tight_layout()
    plt.show()

    print(f"MSE (with Zhukov correction): {mse_zk:.4f}")
    print(f"MAE (with Zhukov correction): {mae_zk:.4f}")
    print(f"RMSE (with Zhukov correction): {rmse_zk:.4f}")
    print(f"R-squared (with Zhukov correction): {r2_zk:.4f}")
    print(f"MAPE (with Zhukov correction): {mape_zk:.4f}")

    print(f"MSE (without Zhukov correction): {mse_no_zk:.4f}")
    print(f"MAE (without Zhukov correction): {mae_no_zk:.4f}")
    print(f"RMSE (without Zhukov correction): {rmse_no_zk:.4f}")
    print(f"R-squared (without Zhukov correction): {r2_no_zk:.4f}")
    print(f"MAPE (without Zhukov correction): {mape_no_zk:.4f}")

def main():
    print("Добро пожаловать в симулятор модели Ижикевича!")
    print("Пожалуйста, введите параметры (оставьте пустым для значений по умолчанию):")
    a = input(f"Параметр a [{default_params['a']}]: ") or default_params['a']
    b = input(f"Параметр b [{default_params['b']}]: ") or default_params['b']
    c = input(f"Параметр c [{default_params['c']}]: ") or default_params['c']
    d = input(f"Параметр d [{default_params['d']}]: ") or default_params['d']
    t_max = float(input("Время симуляции (t_max): "))
    dt = float(input("Шаг времени (dt): "))
    v_threshold = float(input("Порог спайка (v_threshold): "))

    params = {
        'a': float(a),
        'b': float(b),
        'c': float(c),
        'd': float(d)
    }

    I_ext = get_external_current(t_max, dt)
    compare_models(params, t_max, dt, v_threshold, I_ext)


if __name__ == "__main__":
    main()