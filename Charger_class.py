from Vehicle_class import Vehicle
import numpy as np
import math

class Charger:
    # Class variables (shared by all in class)
    
    def __init__(self):
        # Variables unique to each instance *Be sure to update info()*
        self.occupied = False
        self.current_charging_rate = 0
        self.load = 0
        self.current_vehicle = Vehicle()
        self.maximum_load = 1700
        self.current_time = 0
        self.load_log = [(0.0,0.0)]
        self.name = 'NONE'
        self.DC = False
        
        
    def add_vehicle(self,EV):
        # default for now
        self.current_charging_rate = 0
        if self.occupied:
            print("Cannot add vehicle. Charger is currently occupied by vehicle {}.".format(self.current_vehicle.index))
        else:
            self.occupied = True
            self.current_vehicle = EV
            
    def remove_vehicle(self):
        if self.occupied:
            self.occupied = False
            self.current_charging_rate = 0
            self.current_time = self.current_vehicle.current_time
            self.update_load()
            return self.current_vehicle
        else:
            print("No vehicle to remove")
            
    def update_load(self):
        if self.occupied:
            self.load = self.current_charging_rate / self.current_vehicle.charging_efficiency
        else:
            self.load = 0
        
        if self.current_time == self.load_log[len(self.load_log)-1][0]:
            self.load_log[len(self.load_log)-1] = (self.current_time,self.load)
        else:
            self.load_log.append((self.current_time,self.load))
        
    def charge(self,time):      # time is in seconds
        # Test initial log
        self.update_load()
        self.current_vehicle.update_log()
        # Check for vehicle presence
        if not self.occupied:
            print("No vehicle to charge")
            return
        # Disconnect already full vehicle
        if self.current_vehicle.battery_SOC >= 100:
            # Update vehicle information
            self.current_vehicle.update_SOC()
            self.current_vehicle.current_time += time
            self.current_vehicle.update_log()
            
            # Update charger information
            self.current_time = self.current_vehicle.current_time
            self.remove_vehicle()
            self.update_load()
            return
        
        # Set DC charging rate based on SOC
        if self.DC:
            self.set_DC_charge_rate()
        else:
            # Check if AC charging is at the SOC threshold for constant Voltage mode for its battery type
            if ((self.current_vehicle.battery_type == 0 and self.current_vehicle.battery_SOC > 93) or (self.current_vehicle.battery_type == 1 and self.current_vehicle.battery_SOC > 96) or (self.current_vehicle.battery_type == 2 and self.current_vehicle.battery_SOC > 76.1)):
                self.set_AC_charge_rate()
            
            # Set/check charging rate (nonzero/negative, doesn't exceed vehicle parameters, doesn't exceed charger parameters)
            if (self.current_charging_rate <= 0) or (self.current_charging_rate > self.current_vehicle.maximum_charge_rate) or (self.current_charging_rate > (self.maximum_load * self.current_vehicle.charging_efficiency)):
                # Default to maximum if not set
                self.current_charging_rate = min(self.current_vehicle.maximum_charge_rate, (self.maximum_load * self.current_vehicle.charging_efficiency))
        
        # Charge battery
        # Check for overcharge
        if (self.current_vehicle.battery_capacity + ((self.current_charging_rate * time / 3600) / 1000)) > self.current_vehicle.battery_size:
            # Charge up to full over given interval
            remaining_charge = self.current_vehicle.battery_size - self.current_vehicle.battery_capacity #remaining charge in kWh
            self.current_charging_rate = remaining_charge / (time/3600)
            self.current_vehicle.battery_capacity = self.current_vehicle.battery_size
        # Charge at current rate
        else:
            self.current_vehicle.battery_capacity += ((self.current_charging_rate * time / 3600) / 1000)
        
        # Update vehicle information
        self.current_vehicle.update_SOC()
        self.current_vehicle.current_time += time
        self.current_vehicle.update_log()
        
        # Update charger information
        self.current_time = self.current_vehicle.current_time
        self.update_load()
        
    def set_DC_charge_rate(self):
        charging_curves = [[[]]]
        # NMC
        charging_curves[0][0] = [(0,0),(100,0)]
        charging_curves[0].extend([[(0,0.917),(4,1.048),(10,1.095),(88,1.25),(93,0.595),(100,0.06)]])
        charging_curves[0].extend([[(0,1.75),(3,2),(10,2.143),(78.5,2.417),(93,0.595),(100,0.06)]])
        charging_curves[0].extend([[(0,2.798),(3,3.167),(10,3.393),(67,3.75),(93,0.595),(100,0.06)]])
        # LTO
        charging_curves.append([[(0,0),(100,0)]])
        charging_curves[1].extend([[(0,0.798),(2,0.822),(50,0.966),(64,1.008),(80,1.04),(90,1.071),(96,1.134),(100,0.057)]])
        charging_curves[1].extend([[(0,1.765),(2,1.828),(50,1.975),(60,2.038),(80,2.122),(91,2.227),(100,0.057)]])
        charging_curves[1].extend([[(0,2.647),(2,2.794),(50,2.983),(60,3.109),(80,3.256),(88,3.361),(100,0.085)]])
        charging_curves[1].extend([[(0,3.655),(3,3.782),(50,4.055),(60,4.202),(80,4.391),(86,4.517),(100,0.113)]])
        charging_curves[1].extend([[(0,4.622),(4,4.832),(50,5.168),(60,5.357),(84,5.630),(100,0.063)]])
        # LMO
        charging_curves.append([[(0,0),(100,0)]])
        charging_curves[2].extend([[(0,0.898),(4.4,1.056),(11.3,1.154),(32.4,1.215),(76.1,1.274),(100,0.064)]])
        charging_curves[2].extend([[(0,1.742),(4,2.044),(12,2.249),(55,2.418),(75,1.19),(100,0.064)]])
        charging_curves[2].extend([[(0,2.667),(6,3.246),(11.9,3.436),(37.6,3.628),(70,1.44),(100,0.064)]])
        
        # Charge Profile Peak Power(W / Wh)
        #  c - rate	    LMO(2)  NMC(0)  LTO(1)
        #   0	        0	    0	    0
        #   1	        1.27	1.25	1.13
        #   2	        2.42	2.42	2.23
        #   3	        3.63	3.75	3.36
        #   4	        3.63	3.75	4.52
        #   5	        3.63	3.75	5.63
        #   6	        3.63	3.75	5.63
        ###################################################################################################################################
        # Calculate c (Vehicle maximum charge rate assumed to be associated with DC charging)
        c = self.current_vehicle.maximum_charge_rate / (self.current_vehicle.battery_size * 1000)
        ###################################################################################################################################
        c_tolerance = 0.01  # How close before just using the integer value
        c1 = math.floor(c)
        c2 = math.ceil(c)
        
        # Interpolate P
        # Check for c within c_tolerance of an integer value
        if ((np.isclose(c,c1,atol=c_tolerance)) or (np.isclose(c,c2,atol=c_tolerance))):
            c = round(c)
            for i in range(len(charging_curves[self.current_vehicle.battery_type][c])-1):
                soc1 = charging_curves[self.current_vehicle.battery_type][c][i][0]
                soc2 = charging_curves[self.current_vehicle.battery_type][c][i+1][0]
                if (soc1 <= self.current_vehicle.battery_SOC) and (soc2 >= self.current_vehicle.battery_SOC):
                    break
            pc1 = charging_curves[self.current_vehicle.battery_type][c][i][1]
            pc2 = charging_curves[self.current_vehicle.battery_type][c][i+1][1]
            p = pc2 - ((pc2-pc1)*((soc2-self.current_vehicle.battery_SOC)/(soc2-soc1)))
        
        else:
            # Calculate p1
            for i in range(len(charging_curves[self.current_vehicle.battery_type][c1])-1):
                soc1 = charging_curves[self.current_vehicle.battery_type][c1][i][0]
                soc2 = charging_curves[self.current_vehicle.battery_type][c1][i+1][0]
                if (soc1 <= self.current_vehicle.battery_SOC) and (soc2 >= self.current_vehicle.battery_SOC):
                    break
            pc1 = charging_curves[self.current_vehicle.battery_type][c1][i][1]
            pc2 = charging_curves[self.current_vehicle.battery_type][c1][i+1][1]
            p1 = pc2 - ((pc2-pc1)*((soc2-self.current_vehicle.battery_SOC)/(soc2-soc1)))
            # Calculate p2
            for i in range(len(charging_curves[self.current_vehicle.battery_type][c2])-1):
                soc1 = charging_curves[self.current_vehicle.battery_type][c2][i][0]
                soc2 = charging_curves[self.current_vehicle.battery_type][c2][i+1][0]
                if (soc1 <= self.current_vehicle.battery_SOC) and (soc2 >= self.current_vehicle.battery_SOC):
                    break
            pc1 = charging_curves[self.current_vehicle.battery_type][c2][i][1]
            pc2 = charging_curves[self.current_vehicle.battery_type][c2][i+1][1]
            p2 = pc2 - ((pc2-pc1)*((soc2-self.current_vehicle.battery_SOC)/(soc2-soc1)))
            # Calculate p
            p = p2 - ((p2-p1)*((c2-c)/(c2-c1)))
        
        self.current_charging_rate = p * 1000 * self.current_vehicle.battery_size #Factor of 1000 converts kWh battery to Wh
        
    def set_AC_charge_rate(self):
        # c_rate based on AC charger limits
        c = (self.maximum_load * self.current_vehicle.charging_efficiency) / (self.current_vehicle.battery_size * 1000)
        p1 = 0
        c_tolerance = 0.01
        soc2 = 100
        if self.current_vehicle.battery_type == 0:
            soc1 = 88
            pc1 = 1.25
            pc2 = 0.06
        if self.current_vehicle.battery_type == 1:
            soc1 = 96
            pc1 = 1.134
            pc2 = 0.057
        if self.current_vehicle.battery_type == 2:
            soc1 = 76.1
            pc1 = 1.274
            pc2 = 0.064
        if np.isclose(c,1,atol=c_tolerance):
            p2 = pc2
        else:
            p2 = pc2 - ((pc2-pc1)*((soc2-self.current_vehicle.battery_SOC)/(soc2-soc1)))
        
        p = p2 - ((p2 - p1)*((soc2 - self.current_vehicle.battery_SOC)/(soc2 - soc1)))
        
        self.current_charging_rate = p * 1000 * self.current_vehicle.battery_size
            
        
    def info(self):
        help_string = "------ Charger class variables -----\n"
        help_string = help_string + "occupied: \t\t\t\tboolean indicating the presence of a vehicle in the charger \n"
        help_string = help_string + "current_charging_rate: \tthe rate of energy being delivered to the battery accounting for charging efficiency in Watts \n"
        help_string = help_string + "load: \t\t\t\t\tthe load drawn by the charger in Watts \n"
        help_string = help_string + "current_vehicle: \t\tthe most recent vehicle within the charger\n"
        help_string = help_string + "maximum_load: \t\t\tthe highest output of this charger in Watts\n"
        help_string = help_string + "current_time: \t\t\tseconds that have passed since the start of the simulation\n"
        help_string = help_string + "load_log: \t\t\t\tlist of pairs of with (current_time,load)\n"
        help_string = help_string + "name: \t\t\t\t\tstring to help identify the charger\n"
        help_string = help_string + "DC: \t\t\t\t\tboolean flag indicating if this is a DC fast charger\n"
        
        
        
        help_string = help_string + "\n------ Charger class functions -----\n"
        help_string = help_string + "add_vehicle(EV): \t\ttakes in a Vehicle object (EV) and adds it to the charger if it is unoccupied\n"
        help_string = help_string + "remove_vehicle(): \t\tremoves the current vehicle and returns the Vehicle object\n"
        help_string = help_string + "update_load(): \t\t\tchanges load based on current_charging_rate or sets to 0 if unoccupied; adds to load_log\n"
        help_string = help_string + "charge(time): \t\t\ttakes in an amount of time in seconds (time) and charges the current vehicle according to the defined parameters\n"
        
        print(help_string)
        