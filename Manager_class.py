from Vehicle_class import Vehicle
from Charger_class import Charger

class Manager:
    
    def __init__(self):
        
        self.current_time = 0
        self.load = 0
        self.chargers = []         # Change back to empty lists when done
        self.vehicles = []
        self.to_charge = []
        self.load_log = [(0.0,0.0)]
        
    def simulate(self, interval):
        # Reset load
        self.load = 0
        # Check all managed chargers
        for C in self.chargers:
            if C.occupied:
                if C.current_vehicle.battery_SOC >= 100:
                    C.remove_vehicle()
                    # There are vehicles waiting to charge
                    if self.to_charge:
                        C.add_vehicle(self.to_charge[0])
                        del self.to_charge[0]
                        C.charge(interval)
                else:
                    C.charge(interval)
            # Charger is unoccupied
            else:
                # There are vehicles waiting to charge
                if self.to_charge:
                    C.add_vehicle(self.to_charge[0])
                    del self.to_charge[0]
                    C.charge(interval)
                    
            self.load += C.load
            
        # Update current time and load log
        self.current_time += interval
        if self.current_time == self.load_log[len(self.load_log)-1][0]:
            self.load_log[len(self.load_log)-1] = (self.current_time,self.load)
        else:
            self.load_log.append((self.current_time,self.load))
        