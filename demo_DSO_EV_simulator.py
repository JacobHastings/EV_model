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
    
    flag_TOU = True
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
    EV_Chargers = ev_func.intialize_EV_gld_info(EV_dict, duration)
    
    work_chargers_count = 0
    EV_Manager = ev_func.intialize_EV_Manager(EV_Chargers, work_chargers_count)
    
    # exit
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
    
        cosim_bus_mul = max(DSO_KW_cosim_bus.values) / max(gld_KW_cosim_bus.values)
    
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
                # ecap = [100,100,100]
                # pcap = [50,50,25]
                # storage_bus = [2,2,1]
                # eff = [1,1,1]
                # ecap_out = list(ecap)
                # pcap_out = list(pcap)
                # bus_out = list(storage_bus)
                # eff_out = list(eff)
                # storage_specs = dict()
                # storage_specs['ecap'] = ecap_out
                # storage_specs['pcap'] = pcap_out
                # storage_specs['bus'] = bus_out
                # storage_specs['eff'] = eff_out
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

    
    ###### Buffer to sending out data before the Operational Cycle  ######
    buffer = 5
    last_EV_sim_time = 0
    tnext_EV_sim = 0
    t_EV_sim_interval = 60
    
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
                DAM_constant_load_KW = np.zeros(len(flexibility_profile))
                DAM_constant_load_KVAR = np.zeros(len(flexibility_profile))
                flex_load = np.zeros(len(flexibility_profile))
                Q_values = np.zeros([len(flexibility_profile),blocks])
                Q_val_out = [[0]*blocks]*len(flexibility_profile)
                P_values = np.zeros([len(flexibility_profile),blocks])
                P_val_out = [[0]*blocks]*len(flexibility_profile)
                
                
                DAM_start = current_time + timedelta(seconds=buffer)
                DAM_end = DAM_start + timedelta(seconds = wrapper_config['day_ahead_market']['interval'] - 60)

                DAM_profile_KW = DSO_KW_cosim_bus.loc[DAM_start:DAM_end].resample('3600s').mean() 
                MW_MVAR_factor = case['bus'][cosim_bus-1][2] / case['bus'][cosim_bus-1][3]
                DAM_profile_KVAR = DAM_profile_KW//MW_MVAR_factor
                ##### For GLD Simulations: DAM Bids based on previously recorded GLD Data #####
                if include_gld:
                    DAM_profile_KW = gld_KW_cosim_bus_scaled.loc[DAM_start:DAM_end].resample('3600s').mean() 
                    DAM_profile_KVAR = gld_KVAR_cosim_bus_scaled.loc[DAM_start:DAM_end].resample('3600s').mean() 
                
                

                for t in range(len(flexibility_profile)):
                    DAM_constant_load_KW[t] = DAM_profile_KW.iloc[t] * (1-flexibility_profile[t])
                    DAM_constant_load_KVAR[t] = DAM_profile_KVAR.iloc[t]
                    flex_load[t] = DAM_profile_KW.iloc[t] * flexibility_profile[t]
                    Q_values[t] = np.linspace(0,flex_load[t],blocks)
                    Q_val_out[t] = list(Q_values[t])
                    P_values[t] = np.linspace(max(P_range), min(P_range),blocks)
                    P_val_out[t] = list(P_values[t])
                    
                bid =  dict()
                bid['constant_MW'] = list(DAM_constant_load_KW)
                bid['constant_MVAR'] = list(DAM_constant_load_KVAR)
                bid['P_bid'] = P_val_out
                bid['Q_bid'] = Q_val_out
                
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
                if include_helics:                    
                    for cosim_bus in wrapper_config['cosimulation_bus']:
                        sub_key = [key for key in sub_keys  if ('pcc.' + str(cosim_bus) + '.da_energy.cleared') in key ]
                        sub_object = h.helicsFederateGetInputByTarget(fed, sub_key[0])
                        allocation_raw = h.helicsInputGetString(sub_object)
                        DAM_allocation =  json.loads(allocation_raw)
                        print('DSO: Received cleared DAM values {} for Bus {}'.format(DAM_allocation, cosim_bus))
                        
                        for t in range(len(DAM_allocation['P_clear'])):
                            da_bids_df.loc[len(da_bids_df)] = [current_time - timedelta(seconds=buffer) + timedelta(seconds=t*3600), \
                                                               time_granted - buffer + t*3600, DAM_allocation['P_clear'][t], DAM_allocation['Q_clear'][t]]
                                     
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
                RTM_load_KVAR = DAM_profile_KW//MW_MVAR_factor
                
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
        if time_granted >= tnext_EV_sim and time_granted < duration:
             print('DSO: Current EV simulation Interval - {}'.format(time_granted))  
             ev_sim_interval = tnext_EV_sim - last_EV_sim_time
             
             # Check for Time Of Use Disincentive
             if flag_TOU and (TOU_start <= (round(time_granted)%86400) <= TOU_end):
                 TOU_count = 0
                 for C in EV_Chargers:
                     TOU_count += 1
                     if TOU_count > round(TOU_participation*len(EV_Chargers)):
                         break
                     if C.occupied:
                         C.remove_vehicle()
             else:
                 for C in EV_Chargers:
                     if (not C.occupied) and C.current_vehicle.location == 'HOME':
                         C.add_vehicle(C.current_vehicle)
                         
             EV_Chargers, EV_Manager, next_EV_update_time = ev_func.simulate_EVs(EV_Chargers, EV_Manager, time_granted, ev_sim_interval, duration)
             EV_queue_time.append(time_granted)
             EV_queue_length.append(len(EV_Manager.to_charge))
             
             last_EV_sim_time = copy.deepcopy(time_granted)             
             tnext_EV_sim = min(tnext_EV_sim + t_EV_sim_interval, next_EV_update_time, duration)
             

        
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
                for C in EV_Chargers:
                    Charge_avg = ev_func.average_load_interval(C.load_log, time_granted, t_physics_powerflow_interval)
                    EV_Charger_avg_total += Charge_avg
                    # print("Here", str(EV_Charger_avg_total))
                    if include_helics:
                        pub_key = fed_name + "/" + C.name
                        pub_obj = h.helicsFederateGetPublication(fed, pub_key)
                        h.helicsPublicationPublishDouble(pub_obj, Charge_avg)

                # Iterate through work chargers
                for C in EV_Manager.chargers:
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
        
    
    #%%############################ Plotting ########################################
    Aggregate = False
    if Aggregate:
        plot_time, plot_load = ev_func.agregate_loads(EV_Chargers, int(duration), t_EV_sim_interval)
        plot_time = np.array(plot_time)
        plot_time = plot_time / 3600
        plot_load = np.array(plot_load) / 1000
    
        plot_time_m = []
        plot_load_m = []
        for X in EV_Manager.load_log:
            plot_time_m.append(X[0])
            plot_load_m.append(X[1])
            
        plot_time_m = np.array(plot_time_m) / 3600
        plot_load_m = np.array(plot_load_m) / 1000
    
        plot_time_combined, plot_load_combined = ev_func.combine_loads(plot_time,plot_load,plot_time_m,plot_load_m)
        
        fig1, ax1 = plt.subplots(1, 1, figsize =(10, 6), dpi =120)
        ax1.plot(np.array(plot_time_combined),np.array(plot_load_combined), '-', label = 'All EVs Combined')
        ax1.plot(plot_time_m,plot_load_m, '-.', label = 'Work EV Chargers')  
        ax1.plot(plot_time,plot_load, '-.', label = 'Home EV Chargers')
    
        # EV_output_time, EV_output_load = output_from_gridlabd_v2("EV_charger_output.csv")
        # EV_output_time = EV_output_time/3600
        # EV_output_load = EV_output_load/1000
        # plt.plot(EV_output_time,EV_output_load)
    
        ax1.legend(loc = 'best')
        ax1.set_xlabel("Time (hour)")
        ax1.set_ylabel("Agregated Load (Kw)")
        ax1.grid()
        fig1.show()


    fig2, ax2 = plt.subplots(1, 1, figsize =(10, 6), dpi =120)
    ax2.plot(substation_load_df['timestamp'], substation_load_df['substation_load_real'], '-', label = 'Real Demand (MW)')
    ax2.legend(loc = 'best')
    ax2.set_xlabel("Time (hour)")
    ax2.set_ylabel("Feeder Demand")
    ax2.grid()
    fig2.show()
    
    
    # plt.plot(np.array(queue_time)/3600, queue_length)
    # plt.xlabel("Time (hour)")
    # plt.ylabel("Length of Charging Queue")
    # plt.grid()
    # plt.show()

    # plt.plot(Charge_log_time,Charge_log)
    # EV_output_time, EV_output_load = output_from_gridlabd_v2("EV_charger_output.csv")
    # EV_output_time = EV_output_time/3600
    # EV_output_load = EV_output_load/1000
    # plt.plot(EV_output_time,EV_output_load)

    # labels = ['Home Chargers','GLD']
    # plt.legend(labels)
    # plt.xlabel("Time (hour)")
    # plt.ylabel("Agregated Load (Kw)")
    # plt.grid()
    # plt.show()
    
    
    
