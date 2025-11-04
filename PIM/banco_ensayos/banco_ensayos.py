# -*- coding: utf-8 -*-
"""
BancoEnsayos - proceso de datos desde Excel (multiples hojas)
Se asume que cada hoja contiene un ensayo a una temperatura (tiempo vs fuerza).
Salida:
 - gráficos p vs t por hoja (*.png)
 - gráficos μ vs t por hoja (*.png)
 - gráfico resumen μ_efectivo vs T con ajuste polinomial (grado 2..4) (*.png)
 - resumen CSV con columnas: hoja, temperatura, mu_promedio, mu_std, mu_efectivo, etc.
"""
import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class BancoEnsayos:
    def __init__(self):
        self.resultados = []

    @staticmethod
    def _infer_columns(df):
        cols = [c.lower() for c in df.columns]
        time_candidates = ['time', 'tiempo', 't', 'time_s', 'tiempo_s', 'seconds', 'segundos']
        force_candidates = ['force', 'fuerza', 'p', 'p_n', 'force_n', 'fuerza_n', 'adc', 'lectura']
        time_col = None
        force_col = None
        for c in time_candidates:
            for orig in df.columns:
                if c == orig.lower():
                    time_col = orig
                    break
            if time_col:
                break
        for c in force_candidates:
            for orig in df.columns:
                if c == orig.lower():
                    force_col = orig
                    break
            if force_col:
                break
        if time_col is None or force_col is None:
            if len(df.columns) >= 2:
                if time_col is None:
                    time_col = df.columns[0]
                if force_col is None:
                    force_col = df.columns[1]
        return time_col, force_col

    @staticmethod
    def _extract_temperature_from_sheetname(name):
        m = re.search(r'(-?\d+(\.\d+)?)', str(name))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None

    def procesar_excel(self, ruta_excel,
                       sheet_temps=None, output_dir=None,
                       time_col=None, force_col=None,
                       normal_mass=23.2, normal_mass_unit='kg', g=9.81,
                       force_unit='g'):
        """
        Procesa todas las hojas del Excel y devuelve (df_resumen, output_dir).

        Considera errores de medición:
         - error relativo total de la celda de carga: 0.8% (0.5% lectura + 0.3% carga/descarga)
         - error relativo de temperatura (NTC): 0.3%

        Estos errores se propagan a μ (como parte sistemática) y se incluye en el resumen.
        """
        # parámetros de error (relativos)
        sensor_error_rel = 0.008   # 0.8% total en fuerza
        temp_error_rel = 0.003     # 0.3% en temperatura

        if not os.path.exists(ruta_excel):
            raise FileNotFoundError(f"Archivo no encontrado: {ruta_excel}")

        # convertir masa normal a kg
        if str(normal_mass_unit).lower() in ('g', 'gram', 'gramos'):
            normal_mass_kg = float(normal_mass) / 1000.0
        else:
            normal_mass_kg = float(normal_mass)
        normal_force_n = normal_mass_kg * float(g)

        # leer hojas
        xls = pd.read_excel(ruta_excel, sheet_name=None)
        if not xls:
            raise RuntimeError("No se encontraron hojas en el Excel.")

        # carpeta salida
        if output_dir is None:
            base = os.path.dirname(os.path.abspath(ruta_excel))
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(base, f"outputs_analisis_excel_{ts}")
        os.makedirs(output_dir, exist_ok=True)

        resumen_rows = []

        def _safe_name(s):
            return re.sub(r'[^A-Za-z0-9_\-\.]+', '_', str(s)).strip('_')[:120]

        def _parse_force_series(serie):
            s = serie.astype(str).fillna('').str.strip()
            a = s.str.replace(r'\s+', '', regex=True).str.replace(',', '.', regex=False)
            a = a.str.replace(r'[^0-9eE\.\-\+]', '', regex=True)
            a_num = pd.to_numeric(a, errors='coerce')
            b = s.str.replace(r'\s+', '', regex=True)
            b = b.str.replace(r'\.', '', regex=True)
            b = b.str.replace(',', '.', regex=False)
            b = b.str.replace(r'[^0-9eE\.\-\+]', '', regex=True)
            b_num = pd.to_numeric(b, errors='coerce')
            c = s.str.replace(r'[^0-9eE\.\-\+]', '', regex=True)
            c_num = pd.to_numeric(c, errors='coerce')
            candidates = {'a': a_num, 'b': b_num, 'c': c_num}
            best = None
            best_score = -1
            for k, serc in candidates.items():
                valid = int(serc.notna().sum())
                maxv = float(serc.abs().max(skipna=True)) if valid > 0 else 0.0
                score = valid - 1e-6 * maxv
                if score > best_score:
                    best_score = score
                    best = serc
            return best

        # procesar cada hoja
        for sheet_name, df in xls.items():
            if df is None or df.empty:
                print(f"Hoja '{sheet_name}' vacía, se omite.")
                continue

            tcol = time_col
            pcol = force_col
            if tcol is None or pcol is None:
                inferred_t, inferred_p = self._infer_columns(df)
                tcol = tcol or inferred_t
                pcol = pcol or inferred_p

            if tcol not in df.columns or pcol not in df.columns:
                tcol, pcol = self._infer_columns(df)
            if tcol not in df.columns or pcol not in df.columns:
                print(f"ERROR hoja '{sheet_name}': no se pudo determinar columnas tiempo/fuerza. Se omite hoja.")
                continue

            # parseo robusto
            df = df.copy()
            df[tcol] = pd.to_numeric(df[tcol].astype(str).str.replace(r'[^\d\.\-eE+]','', regex=True), errors='coerce')
            df[pcol] = _parse_force_series(df[pcol])
            df = df.dropna(subset=[tcol, pcol]).reset_index(drop=True)
            if df.empty:
                print(f"Hoja '{sheet_name}': no hay datos numéricos válidos después de limpieza. Se omite.")
                continue

            # convertir fuerza a Newtons
            fu = (force_unit or 'N').lower()
            if fu in ('g', 'gram', 'gramo', 'gramos'):
                df['p_N'] = df[pcol] * 0.00980665
            elif fu in ('kg', 'kgf', 'kilogramo', 'kilogramo-fuerza'):
                df['p_N'] = df[pcol] * 9.80665
            else:
                df['p_N'] = df[pcol].astype(float)

            if df[pcol].isna().sum() > 0:
                print(f"Nota hoja '{sheet_name}': {df[pcol].isna().sum()} valores de fuerza no pudieron convertirse; revisar formato de columna.")

            # graficar p vs t
            safe_sheet = _safe_name(sheet_name)
            png_pvst = os.path.join(output_dir, f"{safe_sheet}_p_vs_t.png")
            plt.figure(figsize=(10, 5))
            plt.plot(df[tcol], df['p_N'], '-k', linewidth=1, label="Fuerza (N)")
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Fuerza (N)')
            plt.title(f"Fuerza vs Tiempo - {sheet_name}")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(png_pvst, dpi=200)
            plt.close()

            # calcular mu(t)
            df['mu_t'] = df['p_N'] / normal_force_n
            df['mu_smooth'] = df['mu_t'].rolling(window=5, center=True, min_periods=1).median()

            # onset detection
            n = len(df)
            head_n = max(1, int(0.10 * n))
            baseline = float(df['mu_smooth'].iloc[:head_n].median(skipna=True))
            peak = float(df['mu_smooth'].max(skipna=True))
            threshold = max(baseline + 0.05 * max(peak - baseline, 0.0), baseline + 0.02)
            above = np.where(df['mu_smooth'].values > threshold)[0]
            if above.size > 0:
                onset_idx = int(above[0])
                onset_time = float(df[tcol].iloc[onset_idx])
            else:
                onset_idx = 0
                onset_time = float(df[tcol].iloc[0])

            mu_segment = df['mu_t'].iloc[onset_idx:].reset_index(drop=True)
            t_segment = df[tcol].iloc[onset_idx:].reset_index(drop=True)

            # inicializar regresión lineal (por compatibilidad)
            slope_mu_t = None
            intercept_mu_t = None
            r2_mu_t = None

            # ajuste polinomial en segmento para estimar pico (usar grado 2..4)
            def _fit_and_find_peak(t_vals, y_vals, deg_min=2, deg_max=4):
                nloc = len(t_vals)
                if nloc < 3:
                    return None
                t_mean = np.mean(t_vals)
                t_scale = (np.max(t_vals) - np.min(t_vals)) if np.max(t_vals) != np.min(t_vals) else 1.0
                t_norm = (t_vals - t_mean) / t_scale
                best = None
                best_r2 = -np.inf
                deg_max_use = min(deg_max, nloc - 1)
                for deg in range(max(2, deg_min), deg_max_use + 1):
                    try:
                        p = np.polyfit(t_norm, y_vals, deg)
                        y_fit = np.polyval(p, t_norm)
                        ss_res = np.sum((y_vals - y_fit) ** 2)
                        ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
                        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
                        dp = np.polyder(p)
                        roots = np.roots(dp)
                        real_roots = roots[np.isclose(roots.imag, 0, atol=1e-6)].real
                        candidates = [np.min(t_norm), np.max(t_norm)]
                        for rrt in real_roots:
                            if (rrt >= np.min(t_norm) - 1e-8) and (rrt <= np.max(t_norm) + 1e-8):
                                candidates.append(rrt)
                        peak_val = None
                        peak_t_norm = None
                        for ct in candidates:
                            val = np.polyval(p, ct)
                            if peak_val is None or val > peak_val:
                                peak_val = float(val)
                                peak_t_norm = float(ct)
                        if peak_val is None:
                            continue
                        peak_t = peak_t_norm * t_scale + t_mean
                        result = {
                            'deg': deg,
                            'coeffs_norm': [float(ci) for ci in p],
                            'r2': float(r2) if np.isfinite(r2) else np.nan,
                            'peak_t': float(peak_t),
                            'peak_y': float(peak_val),
                            't_mean': float(t_mean),
                            't_scale': float(t_scale)
                        }
                        if np.isfinite(r2) and r2 > best_r2:
                            best_r2 = r2
                            best = result
                    except Exception:
                        continue
                return best

            fit_result = _fit_and_find_peak(t_segment.values.astype(float), mu_segment.values.astype(float), deg_min=2, deg_max=4)

            # calcular incertidumbres
            if fit_result is not None:
                # reconstruir ajuste sobre los tiempos del segmento para calcular RMSE
                t_mean_loc = fit_result.get('t_mean', np.mean(t_segment.values.astype(float)))
                t_scale_loc = fit_result.get('t_scale', (np.max(t_segment.values.astype(float)) - np.min(t_segment.values.astype(float))) if np.max(t_segment.values.astype(float)) != np.min(t_segment.values.astype(float)) else 1.0)
                t_norm_vals = (t_segment.values.astype(float) - t_mean_loc) / t_scale_loc
                p_norm = np.array(fit_result['coeffs_norm'], dtype=float)
                y_fit_vals = np.polyval(p_norm, t_norm_vals)
                rmse_fit = float(np.sqrt(np.mean((mu_segment.values.astype(float) - y_fit_vals) ** 2)))
                mu_efectivo = float(fit_result['peak_y'])
                mu_peak_time = float(fit_result['peak_t'])
                mu_efectivo_err = float(np.sqrt((mu_efectivo * sensor_error_rel) ** 2 + (rmse_fit) ** 2))
            else:
                if len(mu_segment) == 0:
                    mu_efectivo = float(df['mu_t'].max())
                    mu_peak_time = float(df[tcol].iloc[df['mu_t'].idxmax()])
                    rmse_fit = float(df['mu_t'].std())
                else:
                    idx_rel = int(np.nanargmax(mu_segment.values))
                    mu_efectivo = float(mu_segment.values[idx_rel])
                    mu_peak_time = float(t_segment.iloc[idx_rel])
                    rmse_fit = float(mu_segment.std()) if not np.isnan(mu_segment.std()) else 0.0
                mu_efectivo_err = float(np.sqrt((mu_efectivo * sensor_error_rel) ** 2 + (rmse_fit) ** 2))

            mu_min_seg = float(mu_segment.min()) if not mu_segment.empty else float(df['mu_t'].min())
            mu_max_seg = float(mu_segment.max()) if not mu_segment.empty else float(df['mu_t'].max())

            if mu_efectivo > 5.0:
                print(f"Advertencia hoja '{sheet_name}': μ efectivo = {mu_efectivo:.2f} > 5.0 — verifique unidades.")

            # graficar mu vs t, marcar onset y pico ajustado
            png_muvst = os.path.join(output_dir, f"{safe_sheet}_mu_vs_t.png")
            plt.figure(figsize=(10, 5))
            plt.plot(df[tcol], df['mu_t'], '-b', linewidth=1, label='μ (datos)')
            plt.axvline(x=onset_time, color='gray', linestyle='--', label=f'Inicio movimiento t={onset_time:.2f}s')
            if fit_result is not None:
                xs = np.linspace(t_segment.min(), t_segment.max(), 400)
                t_mean = fit_result.get('t_mean', np.mean(t_segment.values.astype(float)))
                t_scale = fit_result.get('t_scale', (np.max(t_segment.values.astype(float)) - np.min(t_segment.values.astype(float))) if np.max(t_segment.values.astype(float)) != np.min(t_segment.values.astype(float)) else 1.0)
                xs_norm = (xs - t_mean) / t_scale
                p_norm = np.array(fit_result['coeffs_norm'], dtype=float)
                ys_fit = np.polyval(p_norm, xs_norm)
                plt.plot(xs, ys_fit, '--r', linewidth=1.2, label=f'Fit polin. deg={fit_result["deg"]} (R²={fit_result["r2"]:.3f})')
                plt.scatter([fit_result['peak_t']], [fit_result['peak_y']], color='red', zorder=5, label=f'μ pico ajustado={fit_result["peak_y"]:.3f} @ {fit_result["peak_t"]:.2f}s')
            else:
                plt.scatter([mu_peak_time], [mu_efectivo], color='red', zorder=5, label=f'μ pico datos={mu_efectivo:.3f} @ {mu_peak_time:.2f}s')
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Coeficiente de fricción μ')
            plt.title(f"μ vs Tiempo - {sheet_name}")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(png_muvst, dpi=200)
            plt.close()

            # obtener temperatura promedio o desde nombre
            temp_from_name = self._extract_temperature_from_sheetname(sheet_name)
            temp_col_candidates = [c for c in df.columns if re.search(r'temp|temperat|°c|celsius', c, re.IGNORECASE)]
            temp_avg = None
            if temp_col_candidates:
                for tc in temp_col_candidates:
                    try:
                        col_vals = pd.to_numeric(df[tc], errors='coerce')
                        if col_vals.notna().sum() > 0:
                            temp_avg = float(col_vals.mean())
                            break
                    except Exception:
                        continue
            temperatura_c = float(temp_avg) if temp_avg is not None else temp_from_name
            temp_error_c = float(abs(temperatura_c) * temp_error_rel) if temperatura_c is not None else None

            mu_mean_segment = float(mu_segment.mean()) if not mu_segment.empty else float(df['mu_t'].mean())
            mu_std_segment = float(mu_segment.std()) if not mu_segment.empty else float(df['mu_t'].std())
            n_samples = int(len(mu_segment)) if not mu_segment.empty else int(len(df))

            # incertidumbre del promedio: error estadístico + sistemático (sensor)
            stderr_stat = (mu_std_segment / np.sqrt(n_samples)) if n_samples > 1 else 0.0
            mu_promedio_err = float(np.sqrt(stderr_stat ** 2 + (mu_mean_segment * sensor_error_rel) ** 2))

            resumen_rows.append({
                'hoja': sheet_name,
                'temperatura_c': temperatura_c,
                'temp_error_c': temp_error_c,
                'mu_promedio': mu_mean_segment,
                'mu_std': mu_std_segment,
                'mu_promedio_err': mu_promedio_err,
                'n_muestras': n_samples,
                'mu_efectivo': mu_efectivo,
                'mu_efectivo_err': mu_efectivo_err,
                'mu_min_segment': mu_min_seg,
                'mu_max_segment': mu_max_seg,
                'archivo_p_vs_t': os.path.basename(png_pvst),
                'archivo_mu_vs_t': os.path.basename(png_muvst),
                'fit_deg_segment': fit_result['deg'] if fit_result is not None else None,
                'fit_coeffs_segment': json.dumps(fit_result['coeffs_norm']) if fit_result is not None else None,
                'slope_mu_t': slope_mu_t,
                'intercept_mu_t': intercept_mu_t,
                'r2_mu_t': r2_mu_t,
                'onset_time_s': onset_time
            })

            csv_hoja = os.path.join(output_dir, f"{safe_sheet}_procesado.csv")
            df.to_csv(csv_hoja, index=False)
            print(f"Procesada hoja '{sheet_name}': mu_efectivo={mu_efectivo:.4f} ± {mu_efectivo_err:.4f}, mu_promedio_seg={mu_mean_segment:.4f} ± {mu_promedio_err:.4f} (n={n_samples}), archivos en {output_dir}")

        # resumen DataFrame
        df_resumen = pd.DataFrame(resumen_rows)
        if 'temperatura_c' in df_resumen.columns:
            df_resumen = df_resumen.sort_values(by=['temperatura_c'], na_position='last').reset_index(drop=True)

        # ajuste polinomial global usando mu_efectivo vs temperatura (grados 2..4)
        if not df_resumen.empty:
            temps_all = pd.to_numeric(df_resumen['temperatura_c'], errors='coerce').astype(float)
            mus_all = pd.to_numeric(df_resumen['mu_efectivo'], errors='coerce').astype(float)
            mus_err_all = pd.to_numeric(df_resumen.get('mu_efectivo_err', pd.Series(np.zeros_like(mus_all))), errors='coerce').fillna(0).astype(float)
            temp_errs_all = pd.to_numeric(df_resumen.get('temp_error_c', pd.Series(np.zeros_like(temps_all))), errors='coerce').fillna(0).astype(float)

            mask = ~np.isnan(temps_all) & ~np.isnan(mus_all)
            x_all = temps_all[mask].values
            y_all = mus_all[mask].values
            y_errs = mus_err_all[mask].values if len(mus_err_all) == len(temps_all) else np.zeros_like(y_all)
            x_errs = temp_errs_all[mask].values if len(temp_errs_all) == len(temps_all) else np.zeros_like(x_all)

            png_summary = os.path.join(output_dir, "muef_vs_T_resumen.png")
            plt.figure(figsize=(8, 6))

            # Mostrar SOLO los puntos μ_efectivo (máximo por hoja) con sus errores
            if len(x_all) > 0:
                plt.errorbar(x_all, y_all, xerr=x_errs, yerr=y_errs, fmt='o', capsize=4,
                             color='C0', ecolor='C0', mec='k', label='μ_efectivo (hojas)')

            best_deg = None
            best_p = None
            best_r2 = -np.inf
            fits_info = []
            npts = len(x_all)
            if npts >= 2:
                # grados a probar: preferir 2..4; si solo 2 puntos, usar grado 1
                if npts == 2:
                    degs = [1]
                else:
                    degs = [2, 3, 4] if npts >= 4 else list(range(2, npts))
                for deg in degs:
                    try:
                        p = np.polyfit(x_all, y_all, deg)
                        y_fit = np.polyval(p, x_all)
                        ss_res = np.sum((y_all - y_fit) ** 2)
                        ss_tot = np.sum((y_all - np.mean(y_all)) ** 2)
                        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
                        penalized_r2 = r2 - 0.02 * max(0, deg - 2)  # penalización ligera adicional
                        coeffs = [float(ci) for ci in p]
                        fits_info.append({'deg': deg, 'coeffs': coeffs, 'r2': float(r2), 'pen_r2': float(penalized_r2)})
                        if np.isfinite(penalized_r2) and penalized_r2 > best_r2:
                            best_r2 = penalized_r2
                            best_deg = deg
                            best_p = p
                    except Exception as e:
                        print(f"Warning: ajuste global grado {deg} falló: {e}")
                        continue

                # imprimir info de ajustes probados
                print("\nAjustes polinomiales μ_efectivo(T) probados:")
                for info in fits_info:
                    deg = info['deg']
                    coeffs = info['coeffs']
                    r2 = info['r2']
                    pen = info.get('pen_r2', None)
                    terms = []
                    d = deg
                    for i, c in enumerate(coeffs):
                        power = d - i
                        if abs(c) < 1e-12:
                            continue
                        if power == 0:
                            terms.append(f"{c:.6g}")
                        elif power == 1:
                            terms.append(f"{c:.6g}*T")
                        else:
                            terms.append(f"{c:.6g}*T**{power}")
                    poly_str = " + ".join(terms) if terms else "0"
                    print(f"  grado {deg}: R²={r2:.6f} pen_R²={pen:.6f} -> μ_efectivo(T) = {poly_str}")

                # graficar el mejor ajuste y banda MC de incertidumbre
                if best_p is not None:
                    xs = np.linspace(np.min(x_all), np.max(x_all), 400)
                    ys = np.polyval(best_p, xs)
                    plt.plot(xs, ys, '-r', linewidth=1.5, label=f'Fit polin. deg {best_deg}')

                    # MonteCarlo simple para banda de incertidumbre (perturbar x,y según errores)
                    try:
                        rng = np.random.default_rng(0)
                        nmc = 400
                        preds_mc = []
                        for _ in range(nmc):
                            x_s = x_all + rng.normal(scale=x_errs)
                            y_s = y_all + rng.normal(scale=y_errs)
                            try:
                                p_mc = np.polyfit(x_s, y_s, best_deg)
                                preds_mc.append(np.polyval(p_mc, xs))
                            except Exception:
                                continue
                        if len(preds_mc) > 0:
                            preds_mc = np.vstack(preds_mc)
                            pred_std = preds_mc.std(axis=0)
                            plt.fill_between(xs, ys - pred_std, ys + pred_std, color='r', alpha=0.18, label='Fit ±1σ (MC)')
                    except Exception as e:
                        print(f"Warning: MC incertidumbre falló: {e}")

                    # guardar coeficientes en resumen
                    df_resumen = df_resumen.copy()
                    df_resumen['fit_deg_muef_T'] = best_deg
                    df_resumen['fit_coeffs_muef_T'] = json.dumps([float(ci) for ci in best_p.tolist()])

                    # imprimir polinomio seleccionado de forma legible
                    terms = []
                    d = best_deg
                    for i, c in enumerate(best_p):
                        power = d - i
                        if abs(c) < 1e-12:
                            continue
                        if power == 0:
                            terms.append(f"{float(c):.8g}")
                        elif power == 1:
                            terms.append(f"{float(c):.8g}*T")
                        else:
                            terms.append(f"{float(c):.8g}*T**{power}")
                    best_poly_str = " + ".join(terms) if terms else "0"
                    print(f"\nMejor polinomio global (grado {best_deg}):")
                    print(f"  μ_efectivo(T) = {best_poly_str}")
                else:
                    print("No se obtuvo un ajuste polinomial global válido.")
            else:
                print("No hay suficientes puntos con μ_efectivo válidos para ajuste polinomial (necesita al menos 2).")

            plt.xlabel('Temperatura (°C)')
            plt.ylabel('Coeficiente de fricción μ_efectivo')
            plt.title('μ_efectivo vs Temperatura')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(png_summary, dpi=200)
            plt.close()

            csv_summary = os.path.join(output_dir, "resumen_muef_por_hoja.csv")
            df_resumen.to_csv(csv_summary, index=False)
            print(f"\nResumen guardado: {csv_summary}")
            print(f"Gráfico resumen guardado: {png_summary}")
        else:
            print("No hay hojas procesadas para generar resumen.")

        if 'df_resumen' not in locals():
            df_resumen = pd.DataFrame(columns=['hoja','temperatura_c','mu_promedio','mu_std','n_muestras'])
        self.resultados = df_resumen.to_dict(orient='records')
        return df_resumen, output_dir

if __name__ == "__main__":
    # Ejecutar inmediatamente con ruta por defecto (no pedir al usuario)
    ruta_default = "/Users/carlos/Documents/UdeC/2025/Segundo semetre/PIM/Datos/DE_BE_neumatico.xlsx"
    banco = BancoEnsayos()
    df_resumen, out_dir = banco.procesar_excel(ruta_default, sheet_temps=None)
    print(f"\nProcesamiento finalizado. Salida en carpeta: {out_dir}")