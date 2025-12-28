import math

class Memresistor:
    def __init__(self, m_value=0.01, saturation=100):
        self.m_value = m_value
        self.saturation = saturation
        self.mtmp = 0.0
        self.total_output = []
        self.total_ristor = []

    def reset(self):
        """Reset the internal state of the memresistor."""
        self.total_output.clear()
        self.total_ristor.clear()
        self.mtmp = 0.0

    def get_output(self, input_value):
        """Calculate the output based on the input value."""
        self.mtmp += input_value
        if self.mtmp > self.saturation:
            self.mtmp = self.saturation
        elif self.mtmp < -self.saturation:
            self.mtmp = -self.saturation
        self.total_ristor.append(self.mtmp)
        # print(f"mtmp is {self.mtmp}")
        # print(f"inputValue is {input_value}")
        # print(f"mValue is {self.m_value}")
        output = float((self.m_value * abs(self.mtmp)))
        self.total_output.append(output)
        
        return output
    
    def get_output_for(self, input_value):
        """Calculate the output based on the input value."""
        for i in range(1000):
            self.mtmp += input_value
            if self.mtmp > self.saturation:
                self.mtmp = self.saturation
            elif self.mtmp < -self.saturation:
                self.mtmp = -self.saturation
            self.total_ristor.append(self.mtmp)
            output = self.m_value * abs(self.mtmp)

            self.total_output.append(output)
                   
        print(f"mtmp is {self.mtmp}")
        print(f"inputValue is {input_value}")
        print(f"mValue is {self.m_value}")
        return output

    def get_m_tmp(self, input_value):
        """Update and return the internal state (mtmp)."""
        self.mtmp += self.m_value
        if self.mtmp > self.saturation:
            self.mtmp = self.saturation
        elif self.mtmp < -self.saturation:
            self.mtmp = -self.saturation

        self.total_ristor.append(self.mtmp)
        print(f"mtmp is {self.mtmp}")
        print(f"inputValue is {input_value}")
        print(f"mValue is {self.m_value}")

        output = self.mtmp
        self.total_output.append(output)
        return output

    def get_all_output(self):
        """Get all outputs calculated so far."""
        return self.total_output

    def get_total_ristor(self):
        """Get all ristor values calculated so far."""
        return self.total_ristor

# # Example usage
# if __name__ == "__main__":
#     mem = Memresistor(10, 1000)
#     # print(mem.get_output(10))
#     gain = mem.get_output_for(0.01)
#     # print(gain)
#     # print(mem.get_m_tmp(10))
#     print(mem.get_all_output())
#     # print(mem.get_total_ristor())
