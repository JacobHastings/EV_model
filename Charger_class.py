from Vehicle_class import Vehicle

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
        
        
        
            
            
        
    def info(self):
        help_string = "------ Charger class variables -----\n"
        help_string = help_string + "occupied: \t\t\t\tboolean indicating the presence of a vehicle in the charger \n"
        help_string = help_string + "current_charging_rate: \tthe rate of energy being delivered to the battery accounting for charging efficiency in Watts \n"
        help_string = help_string + "load: \t\t\t\t\tthe load drawn by the charger in Watts \n"
        help_string = help_string + "current_vehicle: \t\tthe most recent vehicle within the charger\n"
        help_string = help_string + "maximum_load: \t\t\tthe highest output of this charger in Watts\n"
        help_string = help_string + "current_time: \t\t\tseconds that have passed since the start of the simulation\n"
        help_string = help_string + "load_log: \t\t\t\tlist of pairs of with (current_time,load)\n"
        
        
        
        help_string = help_string + "\n------ Charger class functions -----\n"
        help_string = help_string + "add_vehicle(EV): \t\ttakes in a Vehicle object (EV) and adds it to the charger if it is unoccupied\n"
        help_string = help_string + "remove_vehicle(): \t\tremoves the current vehicle and returns the Vehicle object\n"
        help_string = help_string + "update_load(): \t\t\tchanges load based on current_charging_rate or sets to 0 if unoccupied; adds to load_log\n"
        help_string = help_string + "charge(time): \t\t\ttakes in an amount of time in seconds (time) and charges the current vehicle according to the defined parameters\n"
        
        print(help_string)
        