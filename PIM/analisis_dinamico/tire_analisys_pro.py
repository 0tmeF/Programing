# -*- coding: utf-8 -*-
"""
SISTEMA INTEGRADO DE ANÁLISIS - Nissan Sentra V16
Versión unificada de f_num_2.py + f_neumaticos.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from config import VEHICLE, TIRE, ANALYSIS

class TireAnalysisSystem:
    def __init__(self):
        self.vehicle = VEHICLE
        self.tire = TIRE
        self.df_track = None
        
    def calculate_tire_forces(self, ax_g, ay_g, az_g, mu):
        """Calcula fuerzas con transferencia de carga"""
        M = self.vehicle['MASA']
        g = self.vehicle['GRAVEDAD']
        a = self.vehicle['DISTANCIA_CG_DELANTERO']
        b = self.vehicle['DISTANCIA_CG_TRASERO']
        h_cg = self.vehicle['ALTURA_CG']
        tf = self.vehicle['TROCHA_DELANTERA']
        tr = self.vehicle['TROCHA_TRASERA']
        
        ax = ax_g * g
        ay = ay_g * g
        w = a + b

        # Cargas estáticas
        FzF0 = M * g * b / w
        FzR0 = M * g * a / w

        # Transferencias
        dF_long = M * ax * h_cg / w
        FzF = FzF0 - dF_long
        FzR = FzR0 + dF_long

        dF_lat_front = (M * ay * h_cg) * (b / w) / tf
        dF_lat_rear = (M * ay * h_cg) * (a / w) / tr

        # Distribución por rueda
        Fz_FL = max(0, FzF / 2 - dF_lat_front / 2)
        Fz_FR = max(0, FzF / 2 + dF_lat_front / 2)
        Fz_RL = max(0, FzR / 2 - dF_lat_rear / 2)
        Fz_RR = max(0, FzR / 2 + dF_lat_rear / 2)

        friction_forces = {
            'FL': mu * Fz_FL, 'FR': mu * Fz_FR,
            'RL': mu * Fz_RL, 'RR': mu * Fz_RR
        }

        return Fz_FL, Fz_FR, Fz_RL, Fz_RR, friction_forces

    def analyze_track_data(self, track_data, mu=None):
        """Analiza datos completos de pista"""
        if mu is None:
            mu = self.tire['COEFICIENTE_FRICCION_BASE']
            
        results = []
        for i in range(len(track_data['time'])):
            ax_g = track_data['ax_g'][i] + ANALYSIS['OFFSET_AX']
            ay_g = track_data['ay_g'][i] + ANALYSIS['OFFSET_AY']
            az_g = track_data['az_g'][i] + ANALYSIS['OFFSET_AZ']
            
            Fz_FL, Fz_FR, Fz_RL, Fz_RR, friction = self.calculate_tire_forces(ax_g, ay_g, az_g, mu)
            
            results.append({
                'time': track_data['time'][i],
                'ax_g': ax_g, 'ay_g': ay_g, 'az_g': az_g,
                'Fz_FL': Fz_FL, 'Fz_FR': Fz_FR, 'Fz_RL': Fz_RL, 'Fz_RR': Fz_RR,
                'F_fric_FL': friction['FL'], 'F_fric_FR': friction['FR'],
                'F_fric_RL': friction['RL'], 'F_fric_RR': friction['RR']
            })
        
        self.df_track = pd.DataFrame(results)
        return self.df_track

    def generate_report(self):
        """Genera reporte del análisis"""
        if self.df_track is None:
            print("No hay datos analizados")
            return
            
        print("📊 REPORTE DE ANÁLISIS")
        print("=" * 40)
        for rueda in ['FL', 'FR', 'RL', 'RR']:
            max_fz = self.df_track[f'Fz_{rueda}'].max()
            max_fric = self.df_track[f'F_fric_{rueda}'].max()
            print(f"{rueda}: Fz_max={max_fz:.1f}N, Fric_max={max_fric:.1f}N")

# Uso principal
if __name__ == "__main__":
    sistema = TireAnalysisSystem()
    print("Sistema de Análisis de Neumáticos - Nissan Sentra V16")