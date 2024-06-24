import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Dict, Tuple

# Default parameters for the Hodgkin-Huxley model
default_params = {
    'g_Na': 150.0,  # Increase sodium conductance to make the system more excitable
    'g_K': 20.0,  # Decrease potassium conductance to reduce repolarization
    'g_L': 0.1,  # Decrease leakage conductance to reduce resting membrane potential
    'E_Na': 55.0,  # Increase sodium reversal potential to make the system more excitable
    'E_K': -70.0,  # Decrease potassium reversal potential to reduce repolarization
    'E_L': -50.0,  # Decrease leakage reversal potential to reduce resting membrane potential
    'C_m': 0.5  # Decrease membrane capacitance to increase the speed of membrane potential changes
}

class HodgkinHuxleyModel:
    def __init__(self, params: Dict[str, float], I_ext: np.ndarray, dt: float):
        self.g_Na = params['g_Na']
        self.g_K = params['g_K']
        self.g_L = params['g_L']
        self.E_Na = params['E_Na']
        self.E_K = params['E_K']
        self.E_L = params['E_L']
        self.C_m = params['C_m']
        self.I_ext = I_ext
        self.dt = dt
        self.t_array = np.arange(0, len(I_ext) * dt, dt)

    def alpha_n(self, V: float) -> float:
        V = np.clip(V, -1000, 1000)  # Clipping to prevent overflow
        if V == -55:
            return 0.1  # Approximation when V == -55
        else:
            return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

    def beta_n(self, V: float) -> float:
        V = np.clip(V, -1000, 1000)  # Clipping to prevent overflow
        return 0.125 * np.exp(-(V + 65) / 80)

    def alpha_m(self, V: float) -> float:
        V = np.clip(V, -1000, 1000)  # Clipping to prevent overflow
        if V == -40:
            return 1.0  # Approximation when V == -40
        else:
            return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

    def beta_m(self, V: float) -> float:
        V = np.clip(V, -1000, 1000)  # Clipping to prevent overflow
        return 4.0 * np.exp(-(V + 65) / 18)

    def alpha_h(self, V: float) -> float:
        V = np.clip(V, -1000, 1000)  # Clipping to prevent overflow
        return 0.07 * np.exp(-(V + 65) / 20)

    def beta_h(self, V: float) -> float:
        V = np.clip(V, -1000, 1000)  # Clipping to prevent overflow
        return 1 / (1 + np.exp(-(V + 35) / 10))

    def model(self, t: float, y: np.ndarray) -> np.ndarray:
        V, n, m, h = y
        g_Na = self.g_Na
        g_K = self.g_K
        g_L = self.g_L
        E_Na = self.E_Na
        E_K = self.E_K
        E_L = self.E_L
        C_m = self.C_m

        # Clip V to prevent overflow
        V = np.clip(V, -100, 100)

        I_Na = g_Na * np.clip(m, 0, 1) ** 3 * np.clip(h, 0, 1) * (V - E_Na)
        I_K = g_K * np.clip(n, 0, 1) ** 4 * (V - E_K)
        I_L = g_L * (V - E_L)
        I_ext_t = np.interp(t, self.t_array, self.I_ext)

        dVdt = (I_ext_t - I_Na - I_K - I_L) / C_m
        dndt = self.alpha_n(V) * (1 - n) - self.beta_n(V) * n
        dmdt = self.alpha_m(V) * (1 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1 - h) - self.beta_h(V) * h

        return [dVdt, dndt, dmdt, dhdt]

    def model_pk2(self, t: float, y: np.ndarray) -> np.ndarray:
        V, n, m, h = y

        dVdt, dndt, dmdt, dhdt = self.model(t, y)

        # Ensure finite values
        dVdt = np.nan_to_num(dVdt)
        dndt = np.nan_to_num(dndt)
        dmdt = np.nan_to_num(dmdt)
        dhdt = np.nan_to_num(dhdt)

        V_pred = V + self.dt * dVdt
        n_pred = n + self.dt * dndt
        m_pred = m + self.dt * dmdt
        h_pred = h + self.dt * dhdt

        dVdt_corr, dndt_corr, dmdt_corr, dhdt_corr = self.model(t + self.dt / 2,
                                                                [(V + V_pred) / 2, (n + n_pred) / 2, (m + m_pred) / 2,
                                                                 (h + h_pred) / 2])

        # Ensure finite values
        dVdt_corr = np.nan_to_num(dVdt_corr)
        dndt_corr = np.nan_to_num(dndt_corr)
        dmdt_corr = np.nan_to_num(dmdt_corr)
        dhdt_corr = np.nan_to_num(dhdt_corr)

        return [dVdt_corr, dndt_corr, dmdt_corr, dhdt_corr]

    def model_pk2_zukov(self, t: float, y: np.ndarray) -> np.ndarray:
        V, n, m, h = y

        dVdt, dndt, dmdt, dhdt = self.model(t, y)

        V_pred = V + self.dt * dVdt
        n_pred = n + self.dt * dndt
        m_pred = m + self.dt * dmdt
        h_pred = h + self.dt * dhdt

        dVdt_corr, dndt_corr, dmdt_corr, dhdt_corr = self.model(t + self.dt / 2,
                                                                [(V + V_pred) / 2, (n + n_pred) / 2, (m + m_pred) / 2,
                                                                 (h + h_pred) / 2])

        dVdt_corr = dVdt + self.dt / 2 * (dVdt_corr - dVdt)
        dndt_corr = dndt + self.dt / 2 * (dndt_corr - dndt)
        dmdt_corr = dmdt + self.dt / 2 * (dmdt_corr - dmdt)
        dhdt_corr = dhdt + self.dt / 2 * (dhdt_corr - dhdt)

        return [dVdt_corr, dndt_corr, dmdt_corr, dhdt_corr]

    def model_pk2_zukov_direct(self, t: float, y: np.ndarray) -> np.ndarray:
        V, n, m, h = y

        dVdt, dndt, dmdt, dhdt = self.model(t, y)

        V_pred = V + self.dt * dVdt
        n_pred = n + self.dt * dndt
        m_pred = m + self.dt * dmdt
        h_pred = h + self.dt * dhdt

        dVdt_corr, dndt_corr, dmdt_corr, dhdt_corr = self.model(t + self.dt,
                                                                [V_pred, n_pred, m_pred, h_pred])

        dVdt_corr = dVdt + self.dt / 2 * (dVdt_corr - dVdt)
        dndt_corr = dndt + self.dt / 2 * (dndt_corr - dndt)
        dmdt_corr = dmdt + self.dt / 2 * (dmdt_corr - dmdt)
        dhdt_corr = dhdt + self.dt / 2 * (dhdt_corr - dhdt)

        return [dVdt_corr, dndt_corr, dmdt_corr, dhdt_corr]

    def simulate(self, method: str = 'RK45') -> Tuple[np.ndarray, np.ndarray]:
        y0 = [-65.0, 0.317, 0.05, 0.6]
        if method == 'RK45':
            sol = solve_ivp(self.model, [self.t_array[0], self.t_array[-1]], y0, t_eval=self.t_array, method='RK45')
        elif method == 'PK2':
            sol = solve_ivp(self.model_pk2, [self.t_array[0], self.t_array[-1]], y0, t_eval=self.t_array)
        elif method == 'PK2_Zukov':
            sol = solve_ivp(self.model_pk2_zukov, [self.t_array[0], self.t_array[-1]], y0, t_eval=self.t_array)
        elif method == 'PK2_Zukov_Direct':
            sol = solve_ivp(self.model_pk2_zukov_direct, [self.t_array[0], self.t_array[-1]], y0, t_eval=self.t_array)
        else:
            raise ValueError("Unsupported method")

        return sol.t, sol.y[0]

def get_external_current(t_max: float, dt: float) -> np.ndarray:
    t = np.arange(0, t_max, dt)
    I_ext = np.zeros_like(t)
    I_ext[int(1 / dt):int(1.5 / dt)] = 10.0  # Периодический внешний ток
    I_ext[int(3 / dt):int(3.5 / dt)] = 10.0  # Еще один периодический внешний ток
    return I_ext

def compare_models(params: Dict[str, float], t_max: float, dt: float, I_ext: np.ndarray):
    hh_model = HodgkinHuxleyModel(params, I_ext, dt)

    t_rk45, V_rk45 = hh_model.simulate(method='RK45')
    t_pk2, V_pk2 = hh_model.simulate(method='PK2')
    t_pk2_zukov, V_pk2_zukov = hh_model.simulate(method='PK2_Zukov')
    t_pk2_zukov_direct, V_pk2_zukov_direct = hh_model.simulate(method='PK2_Zukov_Direct')

    # Interpolate results to compare with RK45
    V_pk2_interp = np.interp(t_rk45, t_pk2, V_pk2)
    V_pk2_zukov_interp = np.interp(t_rk45, t_pk2_zukov, V_pk2_zukov)
    V_pk2_zukov_direct_interp = np.interp(t_rk45, t_pk2_zukov_direct, V_pk2_zukov_direct)

    # Calculate metrics for PK2 method
    mse_pk2 = np.mean((V_pk2_interp - V_rk45) ** 2)
    mae_pk2 = np.mean(np.abs(V_pk2_interp - V_rk45))
    rmse_pk2 = np.sqrt(mse_pk2)
    r2_pk2 = 1 - (mse_pk2 / np.var(V_rk45))
    mape_pk2 = np.mean(np.abs((V_pk2_interp - V_rk45) / V_rk45)) * 100

    # Calculate metrics for PK2_Zukov method
    mse_pk2_zukov = np.mean((V_pk2_zukov_interp - V_rk45) ** 2)
    mae_pk2_zukov = np.mean(np.abs(V_pk2_zukov_interp - V_rk45))
    rmse_pk2_zukov = np.sqrt(mse_pk2_zukov)
    r2_pk2_zukov = 1 - (mse_pk2_zukov / np.var(V_rk45))
    mape_pk2_zukov = np.mean(np.abs((V_pk2_zukov_interp - V_rk45) / V_rk45)) * 100

    # Calculate metrics for PK2_Zukov_Direct method
    mse_pk2_zukov_direct = np.mean((V_pk2_zukov_direct_interp - V_rk45) ** 2)
    mae_pk2_zukov_direct = np.mean(np.abs(V_pk2_zukov_direct_interp - V_rk45))
    rmse_pk2_zukov_direct = np.sqrt(mse_pk2_zukov_direct)
    r2_pk2_zukov_direct = 1 - (mse_pk2_zukov_direct / np.var(V_rk45))
    mape_pk2_zukov_direct = np.mean(np.abs((V_pk2_zukov_direct_interp - V_rk45) / V_rk45)) * 100

    # Print metrics
    print(f"Method PK2: MSE = {mse_pk2:.4f}, MAE = {mae_pk2:.4f}, RMSE = {rmse_pk2:.4f}, R2 = {r2_pk2:.4f}, MAPE = {mape_pk2:.4f}%")
    print(f"Method PK2 Zukov: MSE = {mse_pk2_zukov:.4f}, MAE = {mae_pk2_zukov:.4f}, RMSE = {rmse_pk2_zukov:.4f}, R2 = {r2_pk2_zukov:.4f}, MAPE = {mape_pk2_zukov:.4f}%")
    print(f"Method PK2 Zukov Direct: MSE = {mse_pk2_zukov_direct:.4f}, MAE = {mae_pk2_zukov_direct:.4f}, RMSE = {rmse_pk2_zukov_direct:.4f}, R2 = {r2_pk2_zukov_direct:.4f}, MAPE = {mape_pk2_zukov_direct:.4f}%")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(t_rk45, V_rk45, label='RK45', color='blue')
    plt.plot(t_pk2, V_pk2_interp, label='PK2', color='red', linestyle='--')
    plt.plot(t_pk2_zukov, V_pk2_zukov_interp, label='PK2 Zukov', color='green', linestyle='-.')
    plt.plot(t_pk2_zukov_direct, V_pk2_zukov_direct_interp, label='PK2 Zukov Direct', color='purple', linestyle=':')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.legend()
    plt.title('Comparison of Hodgkin-Huxley Models')
    plt.show()

def main():
    # Input parameters
    t_max = float(input("Enter the maximum time (ms): "))
    dt = float(input("Enter the time step (ms): "))

    I_ext = get_external_current(t_max, dt)

    compare_models(default_params, t_max, dt, I_ext)


def main():
    # Input parameters
    t_max = float(input("Enter the maximum time (ms): "))
    dt = float(input("Enter the time step (ms): "))

    I_ext = get_external_current(t_max, dt)

    compare_models(default_params, t_max, dt, I_ext)


if __name__ == "__main__":
    main()