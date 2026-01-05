#!/usr/bin/env python3
import threading
import time
import math
import random
from collections import deque
from threading import Lock

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# SIMULACIÓN TÉRMICA DE NEUMÁTICOS + SENSORES MLX (3 sensores)
# Modelo lumped por neumático: dT/dt = (Q_in - Q_conv - Q_rad) / (m*c)
# Q_in: potencia entregada por pistola térmica (W) cuando apunta al neumático
# Q_conv: h*A*(T - Tamb)
# Q_rad: sigma*e*A*(T^4 - Tamb^4)
# Sensor MLX lee temperatura de superficie (with emissivity, noise, view factor).

sigma_SB = 5.670374419e-8  # W/m2/K4

class ThermalTireSimulator(threading.Thread):
    """
    Simula tres zonas térmicas en la banda de rodadura: outer, mid, inner.
    Cada zona tiene su propia temperatura (K). La pistola entrega potencia total
    que se reparte entre zonas con pequeñas oscilaciones (no uniforme). Hay
    acoplamiento térmico entre zonas para simular conducción lateral.
    """
    def __init__(self, name: str,
                 m_zone=3.0,        # masa efectiva por zona (kg)
                 c=1500.0,          # J/kg/K
                 A_zone=0.04,       # área por zona (m2)
                 emissivity=0.95,
                 h_conv=8.0,        # h*A por zona (W/K)
                 amb_base=25.0,
                 sensor_offset=0.0,
                 sample_hz=5.0,
                 k_cond=8.0,        # mayor conducción lateral por defecto (favorece igualación)
                 sweep_period=6.0,  # segundos para completar outer->mid->inner
                 sweep_amp=0.8):    # amplitud del paneo (0..1)
        super().__init__(daemon=True)
        self.name = name
        # per-zone properties
        self.zones = {
            'outer': {'m': float(m_zone), 'A': float(A_zone), 'T': amb_base + 273.15},
            'mid':   {'m': float(m_zone), 'A': float(A_zone), 'T': amb_base + 273.15},
            'inner': {'m': float(m_zone), 'A': float(A_zone), 'T': amb_base + 273.15},
        }
        self.c = float(c)
        self.eps = float(emissivity)
        self.hA = float(h_conv)   # convective*area (W/K) per zone approx
        self.amb_base = float(amb_base)
        self.sensor_offset = float(sensor_offset)
        self.sample_dt = 1.0 / float(sample_hz)
        self._stop = threading.Event()
        self._lock = Lock()
        self.time0 = time.time()

        # heat gun control (external controller sets these)
        self.heat_power_nominal = 1500.0
        self.heat_target_fraction = 0.0
        self.distance = 0.25
        self.view_base = {'outer': 0.5, 'mid': 0.35, 'inner': 0.15}
        # small time-varying modulation for each zone to create alternating peaks
        self.mod_amp = {'outer': 0.35, 'mid': 0.15, 'inner': 0.45}
        self.mod_freq = {'outer': 0.012, 'mid': 0.009, 'inner': 0.018}
        self.mod_phase = {'outer': 0.0, 'mid': 1.2, 'inner': 2.1}
        # lateral conduction between zones (W/K)
        self.k_cond = float(k_cond)
        # sweep (paneo) parameters: period and amplitude (adds time-varying focus)
        self.sweep_period = float(sweep_period)
        self.sweep_amp = float(sweep_amp)
        # stochastic / erratic behaviour parameters
        self.frac_jitter = 0.25       # fractional jitter applied to each zone allocation (±)
        self.burst_rate = 0.6         # expected bursts per second (higher → más frecuentes)
        self.burst_amp = 0.8          # burst adds up to this fraction of P_tot to a single zone
        self.hA_jitter_pct = 0.20     # convective loss jitter fraction (±)
        self.random_seed = None
        if self.random_seed is not None:
            random.seed(self.random_seed)

        # helper for latest measured (per-zone)
        self.latest = {'t': self.time0, 'obj_outer': self.amb_base, 'obj_mid': self.amb_base, 'obj_inner': self.amb_base, 'amb': self.amb_base}

    def stop(self):
        self._stop.set()

    def set_heat_fraction(self, frac: float, distance: float = None):
        self.heat_target_fraction = max(0.0, min(1.0, float(frac)))
        if distance is not None:
            self.distance = float(distance)

    def effective_power_total(self):
        # simple attenuation by distance^2 and clamp
        att = 1.0 / max(0.02, (self.distance ** 2))
        return max(0.0, min(self.heat_power_nominal * self.heat_target_fraction * att, 8000.0))

    def step_thermal(self, dt, t_rel):
        Tamb_K = self.amb_base + 273.15
        P_tot = self.effective_power_total()
        # compute dynamic allocation fractions per zone (non-uniform, oscillatory)
        fracs = {}
        total = 0.0
        # build per-zone allocation including modulation, sweep/pan and stochastic jitter
        for z in self.zones:
            base = self.view_base.get(z, 0.33)
            mod = self.mod_amp.get(z, 0.2) * math.sin(2 * math.pi * self.mod_freq.get(z, 0.01) * t_rel + self.mod_phase.get(z, 0.0))
            # sweep: triangular-like focus that moves outer -> mid -> inner cyclically
            if self.sweep_period and self.sweep_period > 0:
                sp = ((t_rel % self.sweep_period) / self.sweep_period) * 3.0  # 0..3
                idx_map = {'outer': 0.0, 'mid': 1.0, 'inner': 2.0}
                idx = idx_map.get(z, 1.0)
                weight = max(0.0, 1.0 - abs(sp - idx))  # peak when sp ≈ idx
            else:
                weight = 0.0
            # stochastic per-zone jitter (changes every step)
            jitter = 1.0 + random.uniform(-self.frac_jitter, self.frac_jitter)
            f = max(0.0, base * (1.0 + mod + self.sweep_amp * weight) * jitter)
            fracs[z] = f
            total += f
        # normalize and allocate power
        if total <= 0:
            total = 1.0
        # POSSIBLE RANDOM BURST: occasionally add a short burst to a random zone
        burst_zone = None
        burst_multiplier = 1.0
        # probability scaled with dt
        if random.random() < min(1.0, self.burst_rate * dt):
            burst_zone = random.choice(list(self.zones.keys()))
            burst_multiplier = 1.0 + random.uniform(0.2, self.burst_amp)
        for z, info in self.zones.items():
            frac = fracs[z] / total
            P_in_zone = P_tot * frac * 0.9  # some losses before hitting surface
            if z == burst_zone:
                # apply burst energy as multiplicative factor (short-lived)
                P_in_zone *= burst_multiplier
            # convective loss for zone
            Tz = info['T']
            # add small random fluctuation to convective losses (simulates air gusts / positioning)
            hA_local = self.hA * (1.0 + random.uniform(-self.hA_jitter_pct, self.hA_jitter_pct))
            Q_conv = hA_local * (Tz - Tamb_K)
            # radiative losses (approx)
            Q_rad = sigma_SB * self.eps * info['A'] * (Tz**4 - Tamb_K**4)
            # conduction exchange with neighbors
            Q_cond = 0.0
            # neighbors
            if z == 'outer':
                Tn = self.zones['mid']['T']
                Q_cond = self.k_cond * (Tn - Tz)
            elif z == 'inner':
                Tn = self.zones['mid']['T']
                Q_cond = self.k_cond * (Tn - Tz)
            else:  # mid
                Tn1 = self.zones['outer']['T']
                Tn2 = self.zones['inner']['T']
                Q_cond = self.k_cond * ((Tn1 - Tz) + (Tn2 - Tz))
            # extra stochastic diffusion term to emulate non-uniform heat transfer events
            stochastic_diff = random.uniform(-0.5, 0.5) * 0.5
            dT = (P_in_zone - Q_conv - Q_rad + Q_cond + stochastic_diff) * dt / (info['m'] * self.c)
            info['T'] += dT

    def run(self):
        while not self._stop.is_set():
            t_rel = time.time() - self.time0
            dt = self.sample_dt
            # substep for stability
            nsub = max(1, int(round(dt / 0.02)))
            for _ in range(nsub):
                self.step_thermal(dt / nsub, t_rel)
            # measured temperatures per zone (MLX-like), with emissivity and noise
            with self._lock:
                for z, info in self.zones.items():
                    T_C = info['T'] - 273.15
                    measured = T_C * self.eps + (1.0 - self.eps) * self.amb_base
                    measured += random.gauss(0.0, 0.4)  # sensor noise slightly larger
                    # per-zone sensor offset not added here; sensor_views will simulate offsets
                    self.latest[f'obj_{z}'] = measured
                self.latest['amb'] = self.amb_base + random.gauss(0.0, 0.05)
                self.latest['t'] = time.time()
            time.sleep(self.sample_dt)

# CONTROLADOR DE PISTOLA (schedule)
class StageHeatController(threading.Thread):
    """
    Control secuencial: calienta zonas de una sola rueda en orden (inner -> mid -> outer)
    hasta target_temp (medido por los sensores MLX simulados) y mantiene un breve hold
    para permitir transferencia entre bandas. Evita apagados simultáneos.
    """
    def __init__(self, tire: ThermalTireSimulator,
                 zones_seq=('inner', 'mid', 'outer'),
                 target_temp=50.0,
                 heat_frac=0.9,
                 focus_distance=0.20,
                 hold_time=3.0,
                 poll_dt=0.2):
        super().__init__(daemon=True)
        self.tire = tire
        self.zones_seq = list(zones_seq)
        self.target_temp = float(target_temp)
        self.heat_frac = float(heat_frac)
        self.focus_distance = float(focus_distance)
        self.hold_time = float(hold_time)
        self.poll_dt = float(poll_dt)
        self._stop = threading.Event()
        self._done = threading.Event()

    def stop(self):
        self._stop.set()

    def done(self):
        return self._done.is_set()

    def _focus_on(self, zone):
        # set view factors so most power goes to "zone" (small spill to neighbors)
        if zone == 'inner':
            self.tire.view_base = {'outer': 0.05, 'mid': 0.25, 'inner': 0.70}
        elif zone == 'mid':
            self.tire.view_base = {'outer': 0.15, 'mid': 0.70, 'inner': 0.15}
        else:  # outer
            self.tire.view_base = {'outer': 0.70, 'mid': 0.25, 'inner': 0.05}

    def run(self):
        try:
            for zone in self.zones_seq:
                if self._stop.is_set():
                    break
                print(f"\n▶ Iniciando etapa: calentar zona '{zone}' hasta {self.target_temp}°C")
                # enfocar pistola en la zona objetivo y aplicar potencia
                self._focus_on(zone)
                self.tire.set_heat_fraction(self.heat_frac, distance=self.focus_distance)
                # esperar hasta que la medición de la zona alcance target (o se detenga)
                reached = False
                while not self._stop.is_set():
                    with self.tire._lock:
                        meas = self.tire.latest.get(f'obj_{zone}', self.tire.amb_base)
                    if meas >= self.target_temp:
                        reached = True
                        break
                    time.sleep(self.poll_dt)
                if not reached:
                    break
                print(f"  ✓ Zona '{zone}' alcanzó {meas:.1f}°C — manteniendo {self.hold_time}s para transferencia")
                # mantener potencia reducida para permitir transferencia durante hold_time
                self.tire.set_heat_fraction(self.heat_frac * 0.4, distance=self.focus_distance)
                t0 = time.time()
                while (time.time() - t0) < self.hold_time and not self._stop.is_set():
                    time.sleep(self.poll_dt)
                # después del hold, reducir potencia antes de siguiente etapa
                self.tire.set_heat_fraction(0.15, distance=self.focus_distance)
                time.sleep(0.5)
            # etapa final: apagar y marcar done
            self.tire.set_heat_fraction(0.0)
            print("\n✔ Todas las etapas completadas. Control de etapas finalizado.")
        finally:
            self._done.set()
            # asegurar que pistola esté apagada
            try:
                self.tire.set_heat_fraction(0.0)
            except Exception:
                pass

# Crear UN solo neumático simulado (FR) — usar nombres de parámetro correctos
tire_FR = ThermalTireSimulator("FR",
                               m_zone=9.0,    # masa efectiva por zona [kg]
                               c=1500.0,
                               A_zone=0.12,   # área por zona [m2]
                               emissivity=0.95,
                               h_conv=12.0,
                               amb_base=25.0,
                               sensor_offset=0.0,
                               sample_hz=5.0)
tire_FR.start()
# Ajustes: reducir diferencia del mid respecto a outer/inner y hacer calentamiento ~1 min más lento
for z in tire_FR.zones:
    # aumentar masa efectiva por zona para ralentizar el calentamiento
    tire_FR.zones[z]['m'] = 3.5

# reducir potencia nominal para alargar el tiempo hasta objetivo (~+60 s)
tire_FR.heat_power_nominal = 2200.0

# Modulación más agresiva en outer/inner pero mid menos extrema (menos ventaja)
# frecuencias/phasings mantienen alternancia para "competencia" entre bandas
tire_FR.mod_amp = {'outer': 1.05, 'mid': 0.85, 'inner': 1.10}
tire_FR.mod_freq = {'outer': 0.055, 'mid': 0.065, 'inner': 0.075}
tire_FR.mod_phase = {'outer': 0.0, 'mid': 0.9, 'inner': 1.8}

# Sweep (paneo) mantiene alternancia pero un poco más lento para permitir igualación
tire_FR.sweep_amp = 1.0
tire_FR.sweep_period = 5.0

# aumentar conducción lateral para que con el sweep las tres bandas terminen más próximas
tire_FR.k_cond = 4.5

# View factors: mid aún recibe algo más, pero ahora más cercano a las otras bandas
tire_FR.view_base = {'outer': 0.30, 'mid': 0.40, 'inner': 0.30}

# Iniciar controlador por etapas (inner -> mid -> outer)
stage_ctrl = StageHeatController(tire_FR,
                                 zones_seq=('inner', 'mid', 'outer'),
                                 target_temp=50.0,
                                 heat_frac=0.95,
                                 focus_distance=0.20,
                                 hold_time=4.0,
                                 poll_dt=0.15)
stage_ctrl.start()

# Preparar graficado y colas — ahora 3 sensores apuntando la misma rueda FR:
sensor_views = [
    {"name": "FR_outer", "zone": "outer", "eps": 0.94, "offset": -0.2, "noise": 0.25},
    {"name": "FR_mid",   "zone": "mid",   "eps": 0.96, "offset":  0.0, "noise": 0.18},
    {"name": "FR_inner", "zone": "inner", "eps": 0.92, "offset":  0.3, "noise": 0.30},
]
history_len = 600
times = deque(maxlen=history_len)
data_obj = {s["name"]: deque(maxlen=history_len) for s in sensor_views}
data_amb = {s["name"]: deque(maxlen=history_len) for s in sensor_views}

# estilo matplotlib (fallback)
preferred_styles = ['seaborn-darkgrid', 'seaborn', 'ggplot', 'dark_background']
for st in preferred_styles:
    if st in plt.style.available:
        plt.style.use(st)
        break
else:
    plt.style.use('default')

fig, ax = plt.subplots(figsize=(10,5))
lines = {}
for sv in sensor_views:
    name = sv["name"]
    line, = ax.plot([], [], label=f"{name} measured")
    lines[name] = line

ax.set_xlabel("Tiempo (s, relativo)")
ax.set_ylabel("Temperatura (°C)")
ax.legend(loc="upper left")
ax.set_ylim(20, 110)

start_time = time.time()

def finalize_and_show_snapshot(times_snap, data_snap):
    import matplotlib.pyplot as _plt
    _plt.figure(figsize=(9,5))
    for name, series in data_snap.items():
        _plt.plot(times_snap, series, label=name)
    _plt.axhline(90.0, color='k', linestyle='--', label='Threshold 90°C')
    _plt.xlabel("Tiempo (s, relativo)")
    _plt.ylabel("Temperatura medida (°C)")
    _plt.title("Mediciones MLX por bandas (snapshot hasta 90°C)")
    _plt.legend()
    _plt.grid(alpha=0.3)
    png_out = "mlx_snapshot_90C.png"
    _plt.tight_layout()
    _plt.savefig(png_out, dpi=200)
    print(f"\nGráfica final guardada: {png_out}")
    _plt.show()

finished = {'done': False, 't_reached': None, 'snapshot': None}

def update_frame(frame):
    now = time.time()
    times.append(now - start_time)
    # read zone measurements from tire_FR
    with tire_FR._lock:
        st = dict(tire_FR.latest)
    for sv in sensor_views:
        name = sv["name"]
        zone = sv["zone"]
        base_meas = st.get(f'obj_{zone}', tire_FR.amb_base)
        # add sensor-specific offset and noise (already partially included)
        measured = base_meas + sv['offset'] + random.gauss(0.0, sv['noise'])
        data_obj[name].append(measured)
        data_amb[name].append(st.get('amb', tire_FR.amb_base))
        lines[name].set_data(list(times), list(data_obj[name]))
    if len(times) > 1:
        ax.set_xlim(max(0, times[0]), times[-1] + 0.5)
    out = " | ".join([f"{sv['name']}: meas={data_obj[sv['name']][-1]:.1f}°C" for sv in sensor_views])
    print("\r" + out, end="", flush=True)

    # detect threshold crossing (first time any sensor reaches 90°C)
    if not finished['done']:
        current_max = max(data_obj[sv['name']][-1] for sv in sensor_views)
        if current_max >= 90.0:
            finished['done'] = True
            finished['t_reached'] = times[-1]
            # capture snapshot arrays
            times_snap = list(times)
            data_snap = {sv['name']: list(data_obj[sv['name']]) for sv in sensor_views}
            # stop heating and tire thread after brief delay handled in finally
            print(f"\n\nUmbral 90°C alcanzado en t={finished['t_reached']:.2f}s (max={current_max:.2f}°C)")
            finalize_and_show_snapshot(times_snap, data_snap)
    return list(lines.values())

# ensure cleanup stops threads
ani = FuncAnimation(fig, update_frame, interval=200, blit=False)

try:
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    stage_ctrl.stop()
    tire_FR.stop()
    time.sleep(0.2)
    print("\nSimulación terminada.")