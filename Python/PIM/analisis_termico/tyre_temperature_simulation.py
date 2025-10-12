import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class Compound:
    """Clase para representar un compuesto de neumático."""
    name: str
    heat_capacity: float
    thermal_conductivity: float
    ideal_temperature: float

class TyreTemperatureSimulator:
    def __init__(self, config: Dict[str, Any]):
        self.compounds = [Compound(**c) for c in config.get("compounds", [])]
        self.track_temperature = config.get("track_temperature", 30.0)  # degrees Celsius
        self.base_load = config.get("base_load", 1500.0)  # Load in Newtons
        self.race_distance = config.get("race_distance", 50)  # laps
        self.driving_style_factor = config.get("driving_style_factor", 1.0)  # Aggressiveness multiplier

    def simulate_compound(self, compound: Compound) -> Dict[str, List[float]]:
        """Simula el comportamiento térmico de un compuesto específico."""
        heat_capacity = compound.heat_capacity
        thermal_conductivity = compound.thermal_conductivity
        base_temp = self.track_temperature
        ideal_temp = compound.ideal_temperature

        temperatures: List[float] = []
        performance: List[float] = []
        current_temp = base_temp

        for lap in range(1, self.race_distance + 1):
            # Heat generation
            heat_generated = self.base_load * self.driving_style_factor * 0.01

            # Heat dissipation
            heat_dissipated = (current_temp - base_temp) * thermal_conductivity

            # Update temperature
            current_temp += (heat_generated - heat_dissipated) / heat_capacity

            # Clamp temperature within a realistic range
            current_temp = max(base_temp, current_temp)

            # Performance metric (inverse of temperature deviation from ideal)
            performance_metric = max(0.0, 1.0 - abs(current_temp - ideal_temp) / ideal_temp)

            temperatures.append(current_temp)
            performance.append(performance_metric)

        return {
            "temperatures": temperatures,
            "performance": performance
        }

    def analyse_compounds(self) -> Dict[str, Dict[str, List[float]]]:
        """Analiza todos los compuestos configurados."""
        results = {}

        for compound in self.compounds:
            print(f"Simulating {compound.name} compound...")
            results[compound.name] = self.simulate_compound(compound)

        return results

    def plot_results(self, results: Dict[str, Dict[str, List[float]]]) -> None:
        """Genera gráficos de los resultados de la simulación."""
        laps = list(range(1, self.race_distance + 1))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        # Plot temperature profiles
        for compound_name, data in results.items():
            ax1.plot(laps, data["temperatures"], label=f"{compound_name} Temperature")
        
        ax1.set_title("Tyre Temperature Over Race Distance")
        ax1.set_xlabel("Lap")
        ax1.set_ylabel("Temperature (°C)")
        ax1.legend()
        ax1.grid(True)

        # Plot performance metrics
        for compound_name, data in results.items():
            ax2.plot(laps, data["performance"], label=f"{compound_name} Performance")
        
        ax2.set_title("Tyre Performance Over Race Distance")
        ax2.set_xlabel("Lap")
        ax2.set_ylabel("Performance Metric")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    compounds_config = [
        {
            "name": "Soft",
            "heat_capacity": 0.8,
            "thermal_conductivity": 0.03,
            "ideal_temperature": 85
        },
        {
            "name": "Medium",
            "heat_capacity": 1.0,
            "thermal_conductivity": 0.05,
            "ideal_temperature": 90
        },
        {
            "name": "Hard",
            "heat_capacity": 1.2,
            "thermal_conductivity": 0.07,
            "ideal_temperature": 95
        }
    ]

    config = {
        "compounds": compounds_config,
        "track_temperature": 30,
        "base_load": 1500,
        "race_distance": 50,
        "driving_style_factor": 1.2
    }

    simulator = TyreTemperatureSimulator(config)
    results = simulator.analyse_compounds()
    simulator.plot_results(results)