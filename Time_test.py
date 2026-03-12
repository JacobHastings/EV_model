
from Vehicle_class import Vehicle
from Charger_class import Charger

C = Charger()
V = Vehicle()

C.DC = True
V.battery_size = 10
V.maximum_charge_rate = 30000
V.update_capacity()
C.maximum_load = 40000
C.add_vehicle(V)
time = 0
increment = 1
c_rate = 3
V.battery_SOC = 75

V.update_capacity()
V.maximum_charge_rate = c_rate*1000*V.battery_size

while V.battery_SOC < 100:
    C.charge(increment)
    time += increment
    
print(time)
print(time/60)