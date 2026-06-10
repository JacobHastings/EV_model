# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:35:45 2026

@author: Monish Mukherjee (monish.mukherjee@pnnl.gov)
Pacific Northwest National Laboratory
"""

import math
import json
import csv
import numpy as np
import helics as h
import pandas as pd
from datetime import datetime, timedelta
import modify_glm_dso_ev as glm_mod
import os 
import copy
import glmanip
import dso_ev_functions as ev_func
import matplotlib.pyplot as plt
from Vehicle_class import Vehicle
from Charger_class import Charger
from Manager_class import Manager
import random

def create_broker(simulators, port):
    initstring = "--federates=" + str(simulators) + " --port=" + str(port)
    broker = h.helicsCreateBroker("zmq", "", initstring)
    isconnected = h.helicsBrokerIsConnected(broker)
    if isconnected == 1:
        pass

    return broker

def get_storage_specs(wrapper_config):
    specs_path = '../src/storage_config.json'
    with open(specs_path, 'r') as f:
        specs = json.loads(f.read())
    return specs

def get_load_profiles_from_wrapper(wrapper_config_path, wrapper_config, start_date, end_date):

    load_profile_path = wrapper_config['matpower_most_data']['datapath'] + wrapper_config['matpower_most_data']['load_profile_info']['filename']
    input_resolution = wrapper_config['matpower_most_data']['load_profile_info']['resolution']
    input_data_reference_time =  datetime.strptime(wrapper_config['matpower_most_data']['load_profile_info']['starting_time'], '%Y-%m-%d %H:%M:%S')
    start_data_point = int((start_date - input_data_reference_time).total_seconds() / input_resolution)
    end_data_point   = int((end_date - input_data_reference_time).total_seconds() / input_resolution)
    load_profile_data = pd.read_csv(wrapper_config_path + load_profile_path, skiprows = start_data_point+1, nrows=(end_data_point-start_data_point)+1, header=None)
    new_col_idx = load_profile_data.shape[1]
    load_profile_data[new_col_idx] = pd.date_range(start=start_date, end = end_date, freq='5min')
    load_profile_data.set_index(new_col_idx, inplace=True)
    sample_interval = str(wrapper_config['physics_powerflow']['interval'])+'s'
    load_profile_data_pf_interval = load_profile_data.resample(sample_interval).interpolate()

    return load_profile_data_pf_interval

def get_load_profiles_from_gld(gld_demand_file, pf_interval, start_date, end_date):
    
    gld_demand_df = pd.read_csv(gld_demand_file)
    gld_demand_df['# timestamp'] = pd.to_datetime(gld_demand_df['# timestamp'])
    gld_demand_df_range = gld_demand_df[(gld_demand_df['# timestamp'] >= start_date) & (gld_demand_df['# timestamp'] <=end_date)]
    gld_demand_df_range.set_index('# timestamp', inplace=True)
    sample_interval = str(pf_interval)+'s'
    gld_demand_df_range_interval = gld_demand_df_range.resample(sample_interval).interpolate()

    return gld_demand_df_range_interval




def update_helics_config_wrapper(fed_name, wrapper_config, helics_config_DSO_EV):
    
    Transmission_Sim_name = wrapper_config['helics_config']['name']
    wrapper_broker_port =  wrapper_config['helics_config']['broker_port']
    
    helics_config_DSO_EV['broker_port'] = wrapper_broker_port

    if wrapper_config['include_physics_powerflow']:
        
        for bus in wrapper_config['cosimulation_bus']:
            helics_config_DSO_EV['publications'].append({'global': bool(True),
                                               'key': str(helics_config_DSO_EV['name']+ '.pcc.' + str(bus) + '.pq'),
                                               'type': str('complex')
                                               })
            helics_config_DSO_EV['subscriptions'].append({'required': bool(False),
                                               'key': str(Transmission_Sim_name + '.pcc.' + str(bus) + '.pnv'),
                                               'type': str('complex')
                                               })
                                               
    if wrapper_config['include_real_time_market']:   
        for bus in wrapper_config['cosimulation_bus']:                                        
            helics_config_DSO_EV['publications'].append({'global': bool(True),
                                               'key': str(helics_config_DSO_EV['name']+ '.pcc.' + str(bus) + '.rt_energy.bid'),
                                               'type': str('JSON')
                                               })
    
            helics_config_DSO_EV['subscriptions'].append({'required': bool(False),
                                               'key': str(Transmission_Sim_name + '.pcc.' + str(bus) + '.rt_energy.cleared'),
                                               'type': str('JSON')
                                               })
    if wrapper_config['include_day_ahead_market']:   
        for bus in wrapper_config['cosimulation_bus']:                                        
            helics_config_DSO_EV['publications'].append({'global': bool(True),
                                               'key': str(helics_config_DSO_EV['name']+ '.pcc.' + str(bus) + '.da_energy.bid'),
                                               'type': str('JSON')
                                               })
    
            helics_config_DSO_EV['subscriptions'].append({'required': bool(False),
                                               'key': str(Transmission_Sim_name + '.pcc.' + str(bus) + '.da_energy.cleared'),
                                               'type': str('JSON')
                                               })
    if wrapper_config['include_storage']:
        helics_config_DSO_EV['publications'].append({'global': bool(True),
                                           'key': str(helics_config_DSO_EV['name']+ '.pcc.' + 'storage'),
                                           'type': str('JSON')
                                           })


    return helics_config_DSO_EV



if __name__ == "__main__":

    # json_path = '../src/wrapper_config_test.json'
    

    include_gld = True
    include_wrapper = True
    include_helics = True
    
    flag_TOU = False
    TOU_participation = 0.75
    TOU_start = 3600*17     # start 5:00pm
    TOU_end = 3600*22       # end 10:00pm
    
    curr_dir = os.getcwd()
    
    start_date = '2021-08-05 00:00:00' 
    end_date = '2021-08-07 00:00:00'
    
    feeder_name = 'R4_12_47_1'
    broker_port = 50000
    cosim_bus = 5
    cosim_bus_mul = 4021.74
    pf_interval = 60
    ###########################################################################        
    ################## Reading Wrapper Configuration Json File ################
    if include_wrapper: 
        wrapper_config_path = '../MATPOWER-wrapper/src/'
        wrapper_config_path = 'C:/Users/jacob/Documents/MATPOWER-wrapper/src/'
        wrapper_config_filename = wrapper_config_path + 'wrapper_config_v2.json'
        with open(wrapper_config_filename, 'r') as f:
            wrapper_config = json.loads(f.read())
    
        case_path = wrapper_config_path + wrapper_config['matpower_most_data']['datapath'] + wrapper_config['matpower_most_data']['case_name']
        with open(case_path, 'r') as f:
            case = json.loads(f.read())
        
        start_date = wrapper_config['start_time']
        end_date = wrapper_config['end_time']
        wrapper_include_helics = wrapper_config['include_helics']
        cosim_bus = wrapper_config['cosimulation_bus'][0]
        broker_port = wrapper_config['helics_config']['broker_port']
        
        
        
    start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    end_date   = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
    duration   = (end_date - start_date).total_seconds()
    
    
    ###########################################################################
    #################### Initializing EV Chargers from GLD ####################
    basedir = os.getcwd() + '\\' + 'distribution_simulation' + '\\'
    dir_for_glm = basedir + 'Substation_' + feeder_name+ '.glm'
    
    glm_lines = glmanip.read(dir_for_glm,"",buf=[])
    [model,clock,directives,modules,classes] = glmanip.parse(glm_lines)
    
    clock['starttime'] = '\'' + start_date.strftime("%Y-%m-%d %H:%M:%S") + '\''
    clock['stoptime']  = '\'' + end_date.strftime("%Y-%m-%d %H:%M:%S") + '\''
    
    
    model['metrics_collector_writer'] = {'metrics_collector_1': 
                                         {'interval': '300', 
                                          'interim': '7200', 
                                          'filename': 'outputs/sim_metrics_'}}
        
    # Hardcoded temp fix to remove building EVs attached to non-triplex meters
    comm_ev_list = []
    EV_dict = copy.deepcopy(model['evcharger_det'])
    for ev_id in EV_dict:
        if 'meter' in ev_id:
            comm_ev_list.append(ev_id)
            del model['evcharger_det'][ev_id]
    print('DSO: Found {} commercial EVs - Deleting them for now ..'.format(len(comm_ev_list)))

    
    # Set minimum timestep to sim_interval
    timestep_str = '#set minimum_timestep=' + str(60) + ';\n'
    for i, s in enumerate(directives):
        if '#set minimum_timestep=' in s:
            directives[i] = timestep_str
            print(directives[i])
            
    EV_dict = model['evcharger_det']
    EV_dict_mod = {}
    
    vehicle_count = 100
    M = Manager()
    Vehicles = []
    vehicle_c_rating = 2.5
    EV_index = 0
    for ev_name, ev_info in EV_dict.items():
        if EV_index >= vehicle_count:
            break
        
        V = Vehicle()
        V.name =  ev_name
        V.index = EV_index
        EV_index += 1
        
        V.set_day_schedule(duration)
        V.battery_SOC = 70 
        ev_info['battery_SOC'] = V.battery_SOC
        V.battery_size = float(ev_info['mileage_classification']) / float(ev_info['mileage_efficiency'])
        V.update_capacity()
        V.charging_efficiency = float(ev_info['charging_efficiency'])
        V.mileage_efficiency = float(ev_info['mileage_efficiency'])
        V.maximum_charge_rate = V.battery_size * 1000 * vehicle_c_rating   
        
        # V.work_start = float(ev_info['arrival_at_work'])
        # V.work_duration =  float(ev_info['duration_at_work'])
        # V.commute_distance = float(ev_info['travel_distance'])
        
        V.work_start = int(ev_info['arrival_at_work']) // 100
        V.work_start += (( int(ev_info['arrival_at_work']) % 100 ) / 60)
        V.work_duration = float(ev_info['duration_at_work']) / 3600
        
        
        home_arrival = int(ev_info['arrival_at_home']) // 100
        home_arrival += (( int(ev_info['arrival_at_home']) % 100 ) / 60)
        if home_arrival < V.work_start:
            home_arrival += 24
        V.commute_duration = round(((home_arrival - V.work_start - V.work_duration) * 3600), -1)    # Duration is in seconds
        V.commute_distance = float(ev_info['travel_distance']) / 2

        V.update_log()
        Vehicles.append(V)
        
        EV_dict_mod[ev_name] = copy.deepcopy(ev_info)

    
    M.vehicles = Vehicles
    M.initialize_chargers_from_vehicles()
    M.last_setting_change = np.zeros(len(M.chargers))
        
    M.LMP_est = np.array([10., 10., 10., 10., 10., 10., 20., 20., 20., 20., 20., 20., 20.,
           20., 20., 20., 40., 40., 40., 40., 20., 20., 20., 20.])
    
    M.obj_SOC = 1
    M.obj_price = 1
    M.obj_avg = 1
    M.obj_diff = 1
    flag_controlled_charging = True

    ###########################################################################
    ########## Adding HELICS configurations for the DSO-EV simulator ##########
    fed_name = 'DSO_EV_sim'
    no_of_federates = 1
    if include_helics: 
        
        helics_config_DSO_EV = {}
        helics_config_DSO_EV['name'] = fed_name
        # helics_config_DSO_EV['coreInit'] = '--federates=1'
        helics_config_DSO_EV['coreName'] = fed_name + '_Federate'
        helics_config_DSO_EV['coreType'] = 'zmq'
        helics_config_DSO_EV['broker_port'] = str(broker_port)
        helics_config_DSO_EV['loglevel'] = 'warning'
        helics_config_DSO_EV['publications'] = [];
        helics_config_DSO_EV['subscriptions'] = [];
        
        ######### Reading Wrapper Config and adding Wrapper-based Pub/Subs #######
        if include_wrapper: 
            helics_config_DSO_EV = update_helics_config_wrapper(fed_name, wrapper_config, helics_config_DSO_EV)
            no_of_federates += 1
        
        ######### Reading GLM file and Replacing EVs with Load Objects #######
        if include_gld: 
            model['evcharger_det'] = copy.deepcopy(EV_dict_mod)
            model, helics_config_DSO_EV, helics_config_gld = glm_mod.update_GLD_EVs_with_Loads(model, helics_config_DSO_EV, include_helics)
            with open(basedir + "helics_config_gld.json", 'w') as fp1:
                json.dump(helics_config_gld, fp1, indent=4)            
            ########## Write out glm file with EVs replaced by loads ##########
            ofn = dir_for_glm.replace('.glm', '_cosim.glm')
            glmanip.write(ofn, model, clock, directives, modules, classes)
            no_of_federates += 1
        
    
        helics_config_DSO_EV_filename = f"{fed_name}_helics_config.json"
        with open(helics_config_DSO_EV_filename, 'w') as fp2:
            json.dump(helics_config_DSO_EV, fp2, indent=4) 
               
        

    ##### Getting Load Profiles from input Matpower data #####        
    if include_gld:
        gld_demand_file = basedir + 'Substation_' + feeder_name + '_demand.csv'
        gld_demand_profile = get_load_profiles_from_gld(gld_demand_file, pf_interval, start_date, end_date)
        gld_KW_cosim_bus = gld_demand_profile['distribution_load_real']/1e6
        gld_KVAR_cosim_bus = gld_demand_profile['distribution_load_reac']/1e6
        
    if include_wrapper:
        DSO_kW_profiles = get_load_profiles_from_wrapper(wrapper_config_path, wrapper_config, start_date, end_date)
        DSO_KW_cosim_bus = DSO_kW_profiles[cosim_bus]
        if include_gld:
            cosim_bus_mul = max(DSO_KW_cosim_bus.values) / max(gld_KW_cosim_bus.values)
        else:
            cosim_bus_mul = 4215.427
    
    if include_gld:
        gld_KW_cosim_bus_scaled = gld_KW_cosim_bus*cosim_bus_mul
        gld_KVAR_cosim_bus_scaled = gld_KVAR_cosim_bus*cosim_bus_mul
    
    
    # fig0, ax0 = plt.subplots(1, 1, figsize =(10, 6), dpi =120)
    # ax0.plot(DSO_KW_cosim_bus.index, DSO_KW_cosim_bus.values, '-', label = 'ERCOT')
    # ax0.plot(gld_KW_cosim_bus_scaled.index, gld_KW_cosim_bus_scaled.values, '-', label = 'GLD')
    # ax0.legend(loc = 'best')
    # ax0.set_xlabel("Time (hour)")
    # ax0.set_ylabel("Feeder Demand")
    # ax0.grid()
    # fig0.show()
        
    # exit
    
    ##### Setting up HELICS Configuration #####
    print('DSO: HELICS Version {}'.format(h.helicsGetVersion()))
    if include_helics :

        ##### Starting HELICS Broker #####
        broker = create_broker(no_of_federates, broker_port)
        # h.helicsBrokerDisconnect(broker)
        # exit
        
        ##### Registering DSO Federate #####
        fed = h.helicsCreateCombinationFederateFromConfig(helics_config_DSO_EV_filename)
        print('DSO: Registering {} Federate'.format(h.helicsFederateGetName(fed)))

        pubkeys_count = h.helicsFederateGetPublicationCount(fed)
        pub_keys = []
        for pub_idx in range(pubkeys_count):
            pub_object = h.helicsFederateGetPublicationByIndex(fed, pub_idx)
            pub_keys.append(h.helicsPublicationGetName(pub_object))
        print('DSO: {} Federate has {} Publications'.format(h.helicsFederateGetName(fed), pubkeys_count))
    
    
        subkeys_count = h.helicsFederateGetInputCount(fed)
        sub_keys = []
        for sub_idx in range(subkeys_count):
            sub_object = h.helicsFederateGetInputByIndex(fed, sub_idx)
            sub_keys.append(h.helicsSubscriptionGetTarget(sub_object))
        print('DSO: {} Federate has {} Inputs'.format(h.helicsFederateGetName(fed), subkeys_count))
    
        # print(pub_keys)
        # print(sub_keys)
    
        #####  Entering Execution for DSO Federate #####
        status = h.helicsFederateEnterExecutingMode(fed)
        print('DSO: Federate {} Entering execution'.format(h.helicsFederateGetName(fed)))
        
        if include_wrapper:
            if wrapper_config['include_storage']:
                temp_specs = get_storage_specs(wrapper_config)
                specs_raw = json.dumps(temp_specs)
                pub_key = [key for key in pub_keys  if ('pcc.' + 'storage') in key ]
                pub_object = h.helicsFederateGetPublication(fed, pub_key[0])
                status = h.helicsPublicationPublishString(pub_object, specs_raw)
                print('DSO: Published Storage Specifications')
            

    
    
    EV_queue_time = []
    EV_queue_length = []
    EV_Charge_log = []
    EV_Charge_log_time = []
    substation_load_df = pd.DataFrame(columns=['timestamp', 'time_granted', 'substation_load_real', 'substation_load_imag'])
    EV_fleet_DAM_schedule_df = pd.DataFrame(columns=['time', 'planned', 'adjusted'])
    
    ###### Buffer to sending out data before the Operational Cycle  ######
    buffer = 10
    last_EV_sim_time = 0
    tnext_EV_sim = 0
    t_EV_sim_interval = 300
    
    if include_gld: 
        tnext_physics_powerflow = pf_interval-buffer
        tnext_physics_powerflow_adjust = pf_interval+buffer
        t_physics_powerflow_interval = pf_interval
        
    if include_wrapper:        
        tnext_day_ahead_market  = 0
        tnext_day_ahead_market_adjust  = buffer
        tnext_real_time_market  = wrapper_config['real_time_market']['interval'] - buffer
        tnext_real_time_market_adjust  = wrapper_config['real_time_market']['interval'] +buffer
        tnext_physics_powerflow = wrapper_config['physics_powerflow']['interval']-buffer
        tnext_physics_powerflow_adjust = wrapper_config['physics_powerflow']['interval']+buffer
        
        t_day_ahead_market_interval = wrapper_config['day_ahead_market']['interval']
        t_real_time_market_interval = wrapper_config['real_time_market']['interval']
        t_physics_powerflow_interval = wrapper_config['physics_powerflow']['interval']
   
        
        
    save_outputs = 1
    

    # duration = 300
    time_granted = -1
    
    flexibility = 0
    flexibility_profile = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    flexibility_profile = list(map(lambda x: x * flexibility, (i for i in flexibility_profile))) ##Multiply entire flex_profile by flex
    blocks = 10
    P_range = np.array([10, 20]) 
    
    if include_wrapper:  
        if wrapper_config['include_day_ahead_market']:
            final_interval = wrapper_config['day_ahead_market']['interval']
            da_bids_df = pd.DataFrame(columns=['timestamp', 'time_granted', 'DAM_P_cleared', 'DAM_Q_cleared'])
        if wrapper_config['include_real_time_market']:
            final_interval = wrapper_config['real_time_market']['interval']
            rt_intervals = int(duration/wrapper_config['real_time_market']['interval'])
            rt_index = 0
            rt_bids_df = pd.DataFrame(columns=['timestamp', 'time_granted', 'RTM_P_cleared', 'RTM_Q_cleared'])
        if wrapper_config['include_physics_powerflow']:
            final_interval = wrapper_config['physics_powerflow']['interval']
            pf_intervals = int(duration/wrapper_config['physics_powerflow']['interval'])
            pf_v_df = pd.DataFrame(columns=['timestamp', 'time_granted', 'V'])  
            pf_index = 0
     
    altered_schedule = np.zeros((24,vehicle_count))
    altered_schedule_fleet = np.zeros(24)

    while time_granted < duration-buffer:

        ################## Determing next helics time request ##################
        next_helics_time = tnext_EV_sim
        if include_gld:
            next_helics_time = min([tnext_physics_powerflow, tnext_physics_powerflow_adjust, next_helics_time])
        
        if include_wrapper:
            if wrapper_config['include_day_ahead_market']:
                next_helics_time = min([tnext_day_ahead_market, tnext_day_ahead_market_adjust, next_helics_time])
            if wrapper_config['include_real_time_market']:
                next_helics_time = min([tnext_real_time_market,tnext_real_time_market_adjust,next_helics_time])
            if wrapper_config['include_physics_powerflow']:
                next_helics_time = min([tnext_physics_powerflow,tnext_physics_powerflow_adjust,next_helics_time])
        
        if include_helics:
            #next_helics_time = min([tnext_physics_powerflow, tnext_real_time_market, tnext_day_ahead_market]);
            time_granted = h.helicsFederateRequestTime(fed, next_helics_time)
        else:
            time_granted = next_helics_time
            
        print('DSO: Requested {}s and got Granted {}s'.format(next_helics_time, time_granted))
        current_time = start_date + timedelta(seconds=time_granted)
        print('DSO: Current Time is {}'.format(current_time))

        # if time_granted > 0:
        #     exit
            

        #######################################################################
        ################## Day Ahead Market Bidding Interval ##################
        if include_wrapper: 
            if time_granted >= tnext_day_ahead_market and wrapper_config['include_day_ahead_market'] and time_granted < duration:
                print('DSO: Current Day Ahead Market Interval - {}'.format(time_granted))  
                
                ##### For GLD Simulations: DAM Bids based on previously recorded GLD Data #####
                DAM_constant_load_KW = np.zeros(len(flexibility_profile))
                DAM_constant_load_KVAR = np.zeros(len(flexibility_profile))
                
                DAM_start = current_time + timedelta(seconds=buffer)
                DAM_end = DAM_start + timedelta(seconds = wrapper_config['day_ahead_market']['interval'] - 60)
                
                DAM_profile_KW = DSO_KW_cosim_bus.loc[DAM_start:DAM_end].resample('3600s').mean() 
                MW_MVAR_factor = case['bus'][cosim_bus-1][2] / case['bus'][cosim_bus-1][3]
                DAM_profile_KVAR = DAM_profile_KW//MW_MVAR_factor

                if include_gld:
                    DAM_profile_KW = gld_KW_cosim_bus_scaled.loc[DAM_start:DAM_end].resample('3600s').mean() 
                    DAM_profile_KVAR = gld_KVAR_cosim_bus_scaled.loc[DAM_start:DAM_end].resample('3600s').mean() 
                
                
                M.initial_optimization(cosim_bus_mul)
                bid =  M.bids_DAM

                ### Write code to  add planned_schedule_fleet data into the planned column of EV_fleet_DAM_schedule_df
                EV_fleet_schedule_df_temp = pd.DataFrame(columns=['time', 'planned', 'adjusted'])
                EV_fleet_schedule_df_temp['time'] = list((DAM_profile_KW.index -  start_date).total_seconds())
                EV_fleet_schedule_df_temp['planned'] = list(M.charge_schedule_fleet.copy())


                print("Currently just using the observed Load Forecast")
                print("*** Update Logic here to demonestrate the impact of DSO-EV handshake Currently ***")
                bid['constant_MW'] = list(DAM_profile_KW)
                bid['constant_MVAR'] = list(DAM_profile_KVAR)
                
                # print(bid)
                bid_raw = json.dumps(bid)
                        
                #####  Publishing current loads for Co-SIM Bus the ISO Simulator #####
                if include_helics:
                    pub_key = [key for key in pub_keys  if ('pcc.' + str(cosim_bus) + '.da_energy.bid') in key ]
                    pub_object = h.helicsFederateGetPublication(fed, pub_key[0])
                    status = h.helicsPublicationPublishString(pub_object, bid_raw)
                    print('DSO: Published Bids for Bus {}'.format(cosim_bus))
                        
                        
                tnext_day_ahead_market  = tnext_day_ahead_market + t_day_ahead_market_interval


        #######################################################################
        ################## Day Ahead Market Bidding Interval ##################
        if include_wrapper: 
            if time_granted >= tnext_day_ahead_market_adjust and wrapper_config['include_day_ahead_market'] and time_granted < duration:
                print('DSO: Current Day Ahead Market Adjust Interval - {}'.format(time_granted))  
                ###### Saving the DA Cleared Price: This can be used to schedule EVs #########
                            
                for cosim_bus in wrapper_config['cosimulation_bus']:
                    if include_helics:        
                        sub_key = [key for key in sub_keys  if ('pcc.' + str(cosim_bus) + '.da_energy.cleared') in key ]
                        sub_object = h.helicsFederateGetInputByTarget(fed, sub_key[0])
                        allocation_raw = h.helicsInputGetString(sub_object)
                        DAM_allocation =  json.loads(allocation_raw)
                        print('DSO: Received cleared DAM values {} for Bus {}'.format(DAM_allocation, cosim_bus))
                        for t in range(len(DAM_allocation['P_clear'])):
                            da_bids_df.loc[len(da_bids_df)] = [current_time - timedelta(seconds=buffer) + timedelta(seconds=t*3600), \
                                                               time_granted - buffer + t*3600, DAM_allocation['P_clear'][t], DAM_allocation['Q_clear'][t]]
                        for i in range(len(DAM_allocation['Q_clear'])):
                            altered_schedule_fleet[i] = -1 * (DAM_allocation['Q_clear'][i]*(10**6)) / (cosim_bus_mul)
                            # altered_schedule_fleet[i] = M.charge_schedule_fleet[i] + (1000*random.uniform(-1*sum(M.low_energy_used[i,:]),sum(M.high_energy_used[i,:])))
                    else:
                        for i in range(24):
                            altered_schedule_fleet[i] = M.charge_schedule_fleet[i] + (1000*random.uniform(-1*sum(M.low_energy_used[i,:]),sum(M.high_energy_used[i,:])))
                    
                    if flag_controlled_charging:
                        M.charge_schedule_fleet = altered_schedule_fleet
                        M.final_optimization()    

                    EV_fleet_schedule_df_temp['adjusted'] = (altered_schedule_fleet.copy())
                    EV_fleet_DAM_schedule_df = pd.concat([EV_fleet_DAM_schedule_df, EV_fleet_schedule_df_temp], ignore_index=True)
                        
                tnext_day_ahead_market_adjust  = tnext_day_ahead_market_adjust + t_day_ahead_market_interval
        
        
        
        #######################################################################
        ################## Real Time Market Bidding Interval ##################
        if include_wrapper: 
        
            if time_granted >= tnext_real_time_market and wrapper_config['include_real_time_market'] and time_granted < duration:
                print('DSO: Current RT Market Interval - {}'.format(time_granted))  
                profile_time = current_time + timedelta(seconds=buffer)
                
                data_idx = DSO_KW_cosim_bus.index[DSO_KW_cosim_bus.index == profile_time]
                RTM_load_KW = DSO_KW_cosim_bus.loc[data_idx].values[0]
                MW_MVAR_factor = case['bus'][cosim_bus-1][2] / case['bus'][cosim_bus-1][3]
                RTM_load_KVAR = RTM_load_KW//MW_MVAR_factor
                
                ##### For GLD Simulations: RTM Bids based on last measured substation demand #####
                if include_gld and include_helics:
                    RTM_load_KW = substation_load_df.iloc[-1]['substation_load_real']*cosim_bus_mul
                    RTM_load_KVAR = substation_load_df.iloc[-1]['substation_load_imag']*cosim_bus_mul
                
                
                RTM_constant_load_KW = RTM_load_KW*(1-flexibility)
                RTM_constant_load_KVAR = RTM_load_KVAR
                flex_load = RTM_constant_load_KW*(flexibility)
                Q_values = np.linspace(0, flex_load, blocks)
                P_values = np.linspace(max(P_range), min(P_range), blocks)
                    
                    
                current_load = complex(RTM_constant_load_KW, RTM_constant_load_KVAR)
                bid =  dict()
                bid['constant_MW'] = RTM_constant_load_KW
                bid['constant_MVAR'] = RTM_constant_load_KVAR
                bid['P_bid'] = list(P_values)
                bid['Q_bid'] = list(Q_values)
                    
                    
                bid_raw = json.dumps(bid)
                
                 #####  Publishing current loads for Co-SIM Bus the ISO Simulator #####
                if include_helics:
                    pub_key = [key for key in pub_keys  if ('pcc.' + str(cosim_bus) + '.rt_energy.bid') in key ]
                    pub_object = h.helicsFederateGetPublication(fed, pub_key[0])
                    status = h.helicsPublicationPublishString(pub_object, bid_raw)
                    print('DSO: Published Bids for Bus {}'.format(cosim_bus))
                    
                tnext_real_time_market = tnext_real_time_market + t_real_time_market_interval       
       
        #######################################################################
        ################ Real Time Market Adjusting Interval ##################
        if include_wrapper: 
            if time_granted >= tnext_real_time_market_adjust and wrapper_config['include_real_time_market'] and time_granted < duration:
                print('DSO: Current RT Market Adjust Interval - {}'.format(time_granted))  
                if include_helics:                    
                    for cosim_bus in wrapper_config['cosimulation_bus']:
                        sub_key = [key for key in sub_keys  if ('pcc.' + str(cosim_bus) + '.rt_energy.cleared') in key ]
                        sub_object = h.helicsFederateGetInputByTarget(fed, sub_key[0])
                        allocation_raw = h.helicsInputGetString(sub_object)
                        allocation =  json.loads(allocation_raw)
                        print('DSO: Received RTM cleared values {} for Bus {}'.format(allocation, cosim_bus))
                        rt_bids_df.loc[len(rt_bids_df)] = [current_time, time_granted, allocation['P_clear'], allocation['Q_clear']]
                        rt_index = rt_index + 1
                        
                tnext_real_time_market_adjust = tnext_real_time_market_adjust + t_real_time_market_interval
        
        
        #######################################################################    
        ################## EV Simulation Adjustment Interval ##################
        ############################################################################################ ev_sim_interval = tnext_EV_sim - last_EV_sim_time
        if time_granted >= tnext_EV_sim and time_granted < duration:
             print('DSO: Current EV simulation Interval - {}'.format(time_granted))  
             ev_sim_interval = tnext_EV_sim - last_EV_sim_time
             
             # Check for Time Of Use Disincentive
             if flag_TOU and (TOU_start <= (round(time_granted)%86400) <= TOU_end):
                 TOU_count = 0
                 for C in M.chargers:
                     TOU_count += 1
                     if TOU_count > round(TOU_participation*len(M.chargers)):
                         break
                     if C.occupied:
                         C.remove_vehicle()
             else:
                 # M.simulate_scheduled_DAM(t_EV_sim_interval)
                 if (time_granted - last_EV_sim_time) > 0:
                     M.simulate_scheduled_DAM(time_granted - last_EV_sim_time)
                         
             # EV_Chargers, EV_Manager, next_EV_update_time = ev_func.simulate_EVs(EV_Chargers, EV_Manager, time_granted, ev_sim_interval, duration)
             # EV_queue_time.append(time_granted)
             # EV_queue_length.append(len(EV_Manager.to_charge))
             
             last_EV_sim_time = copy.deepcopy(time_granted)             
             tnext_EV_sim = min(tnext_EV_sim + t_EV_sim_interval, duration)
             

        
        #######################################################################    
        ######################### Power Flow Interval #########################
        ####### 1) Collect GLD load, 2) Send Scaled GLD load to WRapper #######
        if time_granted >= tnext_physics_powerflow and time_granted < duration:
            print('DSO: Current Physics Power Flow Interval - {}'.format(time_granted))  
            if include_gld and include_helics: 
                for sub_key in sub_keys:
                    if 'distribution_load' in sub_key:
                        sub_object = h.helicsFederateGetInputByTarget(fed, sub_key)
                        substation_load = h.helicsInputGetComplex(sub_object)/1e6
                        
                substation_load_df.loc[len(substation_load_df)] = [current_time, time_granted, substation_load.real, substation_load.imag]
            
            
            if include_wrapper and wrapper_config['include_physics_powerflow']:
                profile_time = current_time + timedelta(seconds=buffer)
                
                data_idx = DSO_KW_cosim_bus.index[DSO_KW_cosim_bus.index == profile_time]
                PF_load_KW = DSO_KW_cosim_bus.loc[data_idx].values[0]
                MW_MVAR_factor = case['bus'][cosim_bus-1][2] / case['bus'][cosim_bus-1][3]
                PF_load_KVAR = PF_load_KW//MW_MVAR_factor
                
                ##### For GLD Simulations: RTM Bids based on last measured substation demand #####
                if include_gld and include_helics:
                    PF_load_KW = substation_load_df.iloc[-1]['substation_load_real']*cosim_bus_mul
                    PF_load_KVAR = substation_load_df.iloc[-1]['substation_load_imag']*cosim_bus_mul
                
                
                if include_helics:       
                    #####  Publishing current loads for Co-SIM Bus the ISO Simulator #####
                    pub_key = [key for key in pub_keys  if ('pcc.' + str(cosim_bus) + '.pq') in key ]
                    pub_object = h.helicsFederateGetPublication(fed, pub_key[0])
                    status = h.helicsPublicationPublishComplex(pub_object, PF_load_KW, PF_load_KVAR)
                    print('DSO: Published {}+{}j demand for Bus {}'.format(PF_load_KW, PF_load_KVAR, cosim_bus))
                        

            tnext_physics_powerflow = tnext_physics_powerflow + t_physics_powerflow_interval
            
        #######################################################################    
        ##################### Power Flow Adjust Interval ######################
        #### 1) Send EV load to GLD, 2) Get Wrapper voltages & send to GLD ####
        if time_granted >= tnext_physics_powerflow_adjust and time_granted < duration:
            print('DSO: Current Physics Power Flow Adjust Interval - {}'.format(time_granted))  
            EV_Charger_avg_total = 0
            if include_gld: 
                # Calculate avg charge rate over last sim interval
                # for C in M.chargers:
                #     Charge_avg = ev_func.average_load_interval(C.load_log, time_granted, t_physics_powerflow_interval)
                #     EV_Charger_avg_total += Charge_avg
                #     # print("Here", str(EV_Charger_avg_total))
                #     if include_helics:
                #         pub_key = fed_name + "/" + C.name
                #         pub_obj = h.helicsFederateGetPublication(fed, pub_key)
                #         h.helicsPublicationPublishDouble(pub_obj, Charge_avg)

                # Iterate through work chargers
                for C in M.chargers:
                    Charge_avg = ev_func.average_load_interval(C.load_log, time_granted, t_physics_powerflow_interval)
                    if include_helics:
                        pub_key = fed_name + "/" + C.name
                        pub_obj = h.helicsFederateGetPublication(fed, pub_key)
                        h.helicsPublicationPublishDouble(pub_obj, Charge_avg)
            
            EV_Charge_log.append(EV_Charger_avg_total/1000)
            EV_Charge_log_time.append(time_granted/3600)
            
            if include_wrapper and wrapper_config['include_physics_powerflow'] and include_helics:
                sub_key = [key for key in sub_keys  if ('pcc.' + str(cosim_bus) + '.pnv') in key ]
                sub_object = h.helicsFederateGetInputByTarget(fed, sub_key[0])
                voltage = h.helicsInputGetComplex(sub_object)
                print('DSO: Received {} Voltage for Bus {}'.format(voltage, cosim_bus))
                pf_v_df.loc[len(pf_v_df)] = [current_time, time_granted, voltage]
                pf_index = pf_index + 1
                
                if include_gld: 
                   ###### placeholder code to send voltages to GLD #########
                   a = 1 
                
                    
            tnext_physics_powerflow_adjust = tnext_physics_powerflow_adjust + t_physics_powerflow_interval

    if include_helics:
        if save_outputs and include_wrapper:
            if wrapper_config['include_day_ahead_market']:
                da_bids_df.to_csv(fed_name + "_DAM_Cleared.csv")
            if wrapper_config['include_real_time_market']:
                #np.savetxt("rt_bids.csv", rt_bids_save, delimiter=",")
                rt_bids_df.to_csv(fed_name + "_RTM_Cleared.csv")
            if wrapper_config['include_physics_powerflow']:
                #np.savetxt("pf_voltages.csv", pf_v_save, delimiter=",")
                pf_v_df.to_csv(fed_name + "_PF_Voltages.csv")
        h.helicsFederateDisconnect(fed)
        # h.helicsBrokerWaitForDisconnect(broker,-1)
        h.helicsBrokerDisconnect(broker)
        h.helicsCloseLibrary()
        
    
    ############################# Plotting ########################################
    
    # manager_df = pd.DataFrame(M.load_log, columns=['time', 'demand'])
    # fig1, ax1 = plt.subplots(1, 1, figsize =(10, 6), dpi =120)
    # ax1.plot(np.array(manager_df.time)/3600,np.array(manager_df.demand), '-', label = 'All EVs - Actual')
    # ax1.plot(EV_fleet_DAM_schedule_df.time/3600, EV_fleet_DAM_schedule_df.planned, '-', label = 'All EVs - Planned')
    # ax1.plot(EV_fleet_DAM_schedule_df.time[24:]/3600, EV_fleet_DAM_schedule_df.planned[24:]+(1000*np.sum(M.high_energy_used,axis=1)), '-', label = 'High')
    # ax1.plot(EV_fleet_DAM_schedule_df.time[24:]/3600, EV_fleet_DAM_schedule_df.planned[24:]-(1000*np.sum(M.low_energy_used,axis=1)),'-', label = 'Low')
    # ax1.plot(EV_fleet_DAM_schedule_df.time/3600, EV_fleet_DAM_schedule_df.adjusted, '-', label = 'All EVs - Adjusted')

    # ax1.legend(loc = 'best')
    # ax1.set_xlabel("Time (hour)")
    # ax1.set_ylabel("Net EV Load (Kw)")
    # ax1.grid()
    # fig1.show()


    # fig2, ax2 = plt.subplots(1, 1, figsize =(10, 6), dpi =120)
    # ax2.plot(substation_load_df['timestamp'], substation_load_df['substation_load_real'], '-', label = 'Real Demand (MW)')
    # ax2.legend(loc = 'best')
    # ax2.set_xlabel("Time (hour)")
    # ax2.set_ylabel("Feeder Demand")
    # ax2.grid()
    # fig2.show()
    

