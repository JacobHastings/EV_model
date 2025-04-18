import math

class Vehicle:
    
    # Class variables (shared by all in class)
    
    def __init__(self):
        # Variables unique to each instance *Be sure to update info()*
        self.battery_capacity = 0
        self.battery_SOC = 0.0
        self.battery_size = 20
        self.maximum_charge_rate = 1700
        self.charging_efficiency = 0.9
        self.index = -1             # -1 for unassigned
        
        self.current_time = 0
        self.commute_duration = 1800
        self.commute_distance = 15.0
        self.work_start = 5
        self.work_duration = 11.5
        self.mileage_efficiency = 3.846     # default 0.26kWh/mile
        self.next_state_change = 0
        self.location = "HOME"
        self.SOC_log = [(0.0,0.0)]
        self.schedule = [(0.0,"HOME")]
        
        
    def update_SOC(self):
        self.battery_SOC = (self.battery_capacity / self.battery_size) * 100
        
    def update_capacity(self):
        self.battery_capacity = (self.battery_SOC / 100) * self.battery_size
        
    def update_log(self):
        if self.current_time == self.SOC_log[len(self.SOC_log)-1][0]:  # Overwrite if current time is same as last entry
            self.SOC_log[len(self.SOC_log)-1] = (self.current_time,self.battery_SOC)
        else:
            self.SOC_log.append((self.current_time,self.battery_SOC))
            
    def set_day_schedule(self,sim_end):
        day = 0
        schedule_time = 0
        self.schedule = [(schedule_time,"HOME")]                                    # Assume sim starts at home for now
        days = math.ceil(sim_end/86400)                                             # Determine how many days to schedule
        while day < days:
            schedule_time = 86400*day                                               # Start at 0:00 for that day
            schedule_time += ((self.work_start * 3600) - self.commute_duration)
            self.schedule.append((schedule_time,"DRIVING_WORK"))
            schedule_time += self.commute_duration
            self.schedule.append((schedule_time,"WORK"))
            schedule_time += (self.work_duration*3600)
            self.schedule.append((schedule_time,"DRIVING_HOME"))
            schedule_time += self.commute_duration
            self.schedule.append((schedule_time,"HOME"))
            day += 1
            
        self.next_state_change = self.schedule[1][0]                                # Assume leaving home is first action
        

        
        
    def info(self):
        help_string = "------ Vehicle class variables -----\n"
        help_string = help_string + "battery_capacity: \t\tcurrent energy stored in the battery in kWh \n"
        help_string = help_string + "battery_SOC: \t\t\tState of charge of the battery as a percentage \n"
        help_string = help_string + "battery_size: \t\t\tmaximum energy stored in the battery in kWh \n"
        help_string = help_string + "maximum_charge_rate:\tthe highest allowed charge rate in Watts \n"
        help_string = help_string + "charging_efficiency: \tcharging efficiency of the vehicle expressed as a decimal \n"
        help_string = help_string + "index: \t\t\t\t\tan integer assigned to this vehicle for a label (-1 indicates it is unset)\n"
        
        help_string = help_string + "current_time: \t\t\tseconds that have passed since the start of the simulation\n"
        help_string = help_string + "commute_duration: \t\ttime to drive between work and home in seconds\n"
        help_string = help_string + "commute_distance: \t\tdistance of drive between work and home in miles\n"
        help_string = help_string + "work_start: \t\t\ttime of day to begin work (24 hour clock; 5 -> 5:00am, 14.5 -> 2:30pm) \n"
        help_string = help_string + "work_duration: \t\t\ttime spent at work in hours\n"
        help_string = help_string + "mileage_efficiency: \tefficiency of the vehicle in miles/kWh\n"
        help_string = help_string + "next_state_change: \t\ttime of the next scheduled change in seconds from start of simulation\n"
        help_string = help_string + "location: \t\t\t\tstring indicating location (HOME,WORK,DRIVING_HOME,DRIVING_WORK)\n"
        help_string = help_string + "SOC_log: \t\t\t\tlist of pairs of with (current_time,battery_SOC)\n"
        help_string = help_string + "schedule: \t\t\t\tlist of pairs of with (time,location) for the day\n"
        
        
        
        help_string = help_string + "\n------ Vehicle class functions -----\n"
        help_string = help_string + "update_SOC(): \t\t\tChanges battery_SOC to match battery_capacity \n"
        help_string = help_string + "update_capacity(): \t\tChanges battery_capacity to match battery_SOC \n"
        help_string = help_string + "update_log(): \t\t\tAdds or alters SOC_log based on current_time and battery_SOC\n"
        help_string = help_string + "set_day_schedule(): \tsets schedule for the current day\n"
        
        
        print(help_string)
        