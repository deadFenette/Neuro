import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# Оптимизированные параметры модели Ходжкина-Хаксли
default_params = {
    'g_Na': 120.0,  # мcм/см^2
    'g_K': 36.0,  # мcм/см^2
    'g_L': 0.3,  # мcм/см^2
    'E_Na': 50.0,  # мВ
    'E_K': -77.0,  # мВ
    'E_L': -54.4,  # мВ
    'C_m': 1.0  # мкФ/см^2
}


class HodgkinHuxleyModel:
    def __init__(self, params: Dict[str, float], I_ext: np.ndarray, dt: float):
        self.params = params
        self.I_ext = I_ext
        self.dt = dt
        self.t_array = np.arange(0, len(I_ext) * dt, dt)

    def alpha_n(self, V):
        return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

    def beta_n(self, V):
        return 0.125 * np.exp(-(V + 65) / 80)

    def alpha_m(self, V):
        return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

    def beta_m(self, V):
        return 4.0 * np.exp(-(V + 65) / 18)

    def alpha_h(self, V):
        return 0.07 * np.exp(-(V + 65) / 20)

    def beta_h(self, V):
        return 1 / (1 + np.exp(-(V + 35) / 10))

    def model(self, t: float, y: np.ndarray) -> np.ndarray:
        V, n, m, h = y
        I_ext_t = np.interp(t, self.t_array, self.I_ext)

        g_Na = self.params['g_Na']
        g_K = self.params['g_K']
        g_L = self.params['g_L']
        E_Na = self.params['E_Na']
        E_K = self.params['E_K']
        E_L = self.params['E_L']
        C_m = self.params['C_m']

        I_Na = g_Na * m ** 3 * h * (V - E_Na)
        I_K = g_K * n ** 4 * (V - E_K)
        I_L = g_L * (V - E_L)

        dVdt = (I_ext_t - I_Na - I_K - I_L) / C_m
        dndt = self.alpha_n(V) * (1 - n) - self.beta_n(V) * n
        dmdt = self.alpha_m(V) * (1 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1 - h) - self.beta_h(V) * h

        return np.array([dVdt, dndt, dmdt, dhdt])

    def model_pk2(self, t: float, y: np.ndarray) -> np.ndarray:
        V, n, m, h = y
        I_ext_t = np.interp(t, self.t_array, self.I_ext)

        g_Na = self.params['g_Na']
        g_K = self.params['g_K']
        g_L = self.params['g_L']
        E_Na = self.params['E_Na']
        E_K = self.params['E_K']
        E_L = self.params['E_L']
        C_m = self.params['C_m']

        I_Na = g_Na * m ** 3 * h * (V - E_Na)
        I_K = g_K * n ** 4 * (V - E_K)
        I_L = g_L * (V - E_L)

        dVdt = (I_ext_t - I_Na - I_K - I_L) / C_m
        dndt = self.alpha_n(V) * (1 - n) - self.beta_n(V) * n
        dmdt = self.alpha_m(V) * (1 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1 - h) - self.beta_h(V) * h

        V_next = V + self.dt * dVdt
        n_next = n + self.dt * dndt
        m_next = m + self.dt * dmdt
        h_next = h + self.dt * dhdt

        I_Na_next = g_Na * m_next ** 3 * h_next * (V_next - E_Na)
        I_K_next = g_K * n_next ** 4 * (V_next - E_K)
        I_L_next = g_L * (V_next - E_L)

        dVdt_next = (I_ext_t - I_Na_next - I_K_next - I_L_next) / C_m
        dndt_next = self.alpha_n(V_next) * (1 - n_next) - self.beta_n(V_next) * n_next
        dmdt_next = self.alpha_m(V_next) * (1 - m_next) - self.beta_m(V_next) * m_next
        dhdt_next = self.alpha_h(V_next) * (1 - h_next) - self.beta_h(V_next) * h_next

        dVdt_corr = (dVdt + dVdt_next) / 2
        dndt_corr = (dndt + dndt_next) / 2
        dmdt_corr = (dmdt + dmdt_next) / 2
        dhdt_corr = (dhdt + dhdt_next) / 2

        return np.array([dVdt_corr, dndt_corr, dmdt_corr, dhdt_corr])

    def model_pk2_zukov(self, t: float, y: np.ndarray) -> np.ndarray:
        V, n, m, h = y
        I_ext_t = np.interp(t, self.t_array, self.I_ext)

        g_Na = self.params['g_Na']
        g_K = self.params['g_K']
        g_L = self.params['g_L']
        E_Na = self.params['E_Na']
        E_K = self.params['E_K']
        E_L = self.params['E_L']
        C_m = self.params['C_m']

        I_Na = g_Na * m ** 3 * h * (V - E_Na)
        I_K = g_K * n ** 4 * (V - E_K)
        I_L = g_L * (V - E_L)

        dVdt = (I_ext_t - I_Na - I_K - I_L) / C_m
        dndt = self.alpha_n(V) * (1 - n) - self.beta_n(V) * n
        dmdt = self.alpha_m(V) * (1 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1 - h) - self.beta_h(V) * h

        V_next = V + self.dt * dVdt
        n_next = n + self.dt * dndt
        m_next = m + self.dt * dmdt
        h_next = h + self.dt * dhdt

        I_Na_next = g_Na * m_next ** 3 * h_next * (V_next - E_Na)
        I_K_next = g_K * n_next ** 4 * (V_next - E_K)
        I_L_next = g_L * (V_next - E_L)

        dVdt_next = (I_ext_t - I_Na_next - I_K_next - I_L_next) / C_m
        dndt_next = self.alpha_n(V_next) * (1 - n_next) - self.beta_n(V_next) * n_next
        dmdt_next = self.alpha_m(V_next) * (1 - m_next) - self.beta_m(V_next) * m_next
        dhdt_next = self.alpha_h(V_next) * (1 - h_next) - self.beta_h(V_next) * h_next

        dVdt_corr = (dVdt + dVdt_next) / 2
        dndt_corr = (dndt + dndt_next) / 2
        dmdt_corr = (dmdt + dmdt_next) / 2
        dhdt_corr = (dhdt + dhdt_next) / 2

        # Коррекция Жукова
        dVdt_corr += self.dt * dVdt_corr * dVdt_next
        dndt_corr += self.dt * dndt_corr * dndt_next
        dmdt_corr += self.dt * dmdt_corr * dmdt_next
        dhdt_corr += self.dt * dhdt_corr * dhdt_next

        return np.array([dVdt_corr, dndt_corr, dmdt_corr, dhdt_corr])

    def simulate(self, method: str = 'RK45') -> Tuple[np.ndarray, np.ndarray]:
        y0 = [-65.0, 0.317, 0.05, 0.6]
        if method == 'RK45':
            sol = solve_ivp(self.model, [self.t_array[0], self.t_array[-1]], y0, t_eval=self.t_array, method='RK45')
        elif method == 'PK2':
            sol = solve_ivp(self.model_pk2, [self.t_array[0], self.t_array[-1]], y0, t_eval=self.t_array, method='RK45')
        elif method == 'PK2_Zukov':
            sol = solve_ivp(self.model_pk2_zukov, [self.t_array[0], self.t_array[-1]], y0, t_eval=self.t_array,
                            method='RK45')
        else:
            raise ValueError("Unsupported method")

        return sol.t, sol.y[0]


def get_external_current(t_max: float, dt: float) -> np.ndarray:
    t = np.arange(0, t_max, dt)
    I_ext = np.zeros_like(t)
    I_ext[int(1 / dt):int(1.5 / dt)] = 10.0  # Периодический внешний ток
    return I_ext


def compare_models(params: Dict[str, float], t_max: float, dt: float, I_ext: np.ndarray):
    hh_model = HodgkinHuxleyModel(params, I_ext, dt)

    t_rk45, V_rk45 = hh_model.simulate(method='RK45')
    t_pk2, V_pk2 = hh_model.simulate(method='PK2')
    t_pk2_zukov, V_pk2_zukov = hh_model.simulate(method='PK2_Zukov')

    V_pk2_interp = np.interp(t_pk2, t_rk45, V_rk45)
    V_pk2_zukov_interp = np.interp(t_pk2_zukov, t_rk45, V_rk45)

    mse_pk2 = np.mean((V_pk2_interp - V_rk45) ** 2)
    mae_pk2 = np.mean(np.abs(V_pk2_interp - V_rk45))
    rmse_pk2 = np.sqrt(mse_pk2)
    r2_pk2 = 1 - (mse_pk2 / np.var(V_rk45))
    mape_pk2 = np.mean(np.abs((V_pk2_interp - V_rk45) / V_rk45)) * 100

    mse_pk2_zukov = np.mean((V_pk2_zukov_interp - V_rk45) ** 2)
    mae_pk2_zukov = np.mean(np.abs(V_pk2_zukov_interp - V_rk45))
    rmse_pk2_zukov = np.sqrt(mse_pk2_zukov)
    r2_pk2_zukov = 1 - (mse_pk2_zukov / np.var(V_rk45))
    mape_pk2_zukov = np.mean(np.abs((V_pk2_zukov_interp - V_rk45) / V_rk45)) * 100

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(t_rk45, V_rk45, label='No PK2 (RK45)')
    axs[0, 0].plot(t_pk2, V_pk2, label='PK2')
    axs[0, 0].plot(t_pk2_zukov, V_pk2_zukov, label='PK2 + Zukov')
    axs[0, 0].set_xlabel('Time (ms)')
    axs[0, 0].set_ylabel('Membrane potential (mV)')
    axs[0, 0].set_title('Membrane potential over time')
    axs[0, 0].legend()

    axs[1, 0].plot(V_rk45, label='No PK2 (RK45)')
    axs[1, 0].plot(V_pk2_interp, label='PK2')
    axs[1, 0].plot(V_pk2_zukov_interp, label='PK2 + Zukov')
    axs[1, 0].set_xlabel('Time (ms)')
    axs[1, 0].set_ylabel('Membrane potential (mV)')
    axs[1, 0].set_title('Interpolated membrane potential')
    axs[1, 0].legend()

    axs[1, 1].bar(['PK2', 'PK2 + Zukov'], [mse_pk2, mse_pk2_zukov])
    axs[1, 1].set_xlabel('Model')
    axs[1, 1].set_ylabel('MSE')
    axs[1, 1].set_title('Mean squared error')

    plt.tight_layout()
    plt.show()

    print(f"MSE (PK2): {mse_pk2:.4f}")
    print(f"MAE (PK2): {mae_pk2:.4f}")
    print(f"RMSE (PK2): {rmse_pk2:.4f}")
    print(f"R-squared (PK2): {r2_pk2:.4f}")
    print(f"MAPE (PK2): {mape_pk2:.4f}")

    print(f"MSE (PK2 + Zukov): {mse_pk2_zukov:.4f}")
    print(f"MAE (PK2 + Zukov): {mae_pk2_zukov:.4f}")
    print(f"RMSE (PK2 + Zukov): {rmse_pk2_zukov:.4f}")
    print(f"R-squared (PK2 + Zukov): {r2_pk2_zukov:.4f}")
    print(f"MAPE (PK2 + Zukov): {mape_pk2_zukov:.4f}")


def main():
    print("Добро пожаловать в симулятор модели Ходжкина-Хаксли!")
    print("Пожалуйста, введите параметры (оставьте пустым для значений по умолчанию):")
    g_Na = input(f"Параметр g_Na [{default_params['g_Na']}]: ") or default_params['g_Na']
    g_K = input(f"Параметр g_K [{default_params['g_K']}]: ") or default_params['g_K']
    g_L = input(f"Параметр g_L [{default_params['g_L']}]: ") or default_params['g_L']
    E_Na = input(f"Параметр E_Na [{default_params['E_Na']}]: ") or default_params['E_Na']
    E_K = input(f"Параметр E_K [{default_params['E_K']}]: ") or default_params['E_K']
    E_L = input(f"Параметр E_L [{default_params['E_L']}]: ") or default_params['E_L']
    C_m = input(f"Параметр C_m [{default_params['C_m']}]: ") or default_params['C_m']
    t_max = float(input("Время симуляции (t_max): "))
    dt = float(input("Шаг времени (dt): "))

    params = {
        'g_Na': float(g_Na),
        'g_K': float(g_K),
        'g_L': float(g_L),
        'E_Na': float(E_Na),
        'E_K': float(E_K),
        'E_L': float(E_L),
        'C_m': float(C_m)
    }

    I_ext = get_external_current(t_max, dt)
    compare_models(params, t_max, dt, I_ext)


if __name__ == "__main__":
    main()