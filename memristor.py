from typing import List

class Memristor:
    def __init__(self, m_value: float = 0.01, saturation: float = 100.0):
        self.m_value = m_value
        self.saturation = saturation
        self.mtmp = 0.0
        
        self.total_output: List[float] = []
        self.total_ristor: List[float] = []

    def reset(self):
        self.total_output.clear()
        self.total_ristor.clear()
        self.mtmp = 0.0

    def _update_and_clamp(self, delta: float):
        self.mtmp += delta
        self.mtmp = max(-self.saturation, min(self.mtmp, self.saturation))
        self.total_ristor.append(self.mtmp)

    def get_output(self, input_value: float) -> float:
        self._update_and_clamp(input_value)
        
        output = float(self.m_value * abs(self.mtmp))
        self.total_output.append(output)
        
        return output
    
    def simulate_steps(self, input_value: float, steps: int = 1000) -> float:
        output = 0.0
        for _ in range(steps):
            output = self.get_output(input_value)
        return output

    def get_state_direct(self, input_value: float) -> float:
        self._update_and_clamp(self.m_value)
        
        output = self.mtmp
        self.total_output.append(output)
        return output

    def get_all_output(self) -> List[float]:
        return self.total_output

    def get_total_ristor(self) -> List[float]:
        return self.total_ristor