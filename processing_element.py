'''!
@brief This file contains the process elements and their attributes.
'''
import simpy
import copy

import common                                                                           # The common parameters used in DASH-Sim are defined in common_parameters.py
import DTPM_power_models
import DASH_Sim_utils
import DTPM_policies

class PE:
    '''!
    A processing element (PE) is the basic resource that defines the simpy processes.
    '''
    def __init__(self, env, type, name, ID, cluster_ID, capacity,throughput,power):
        '''!
        @param env: Pointer to the current simulation environment
        @param type: Type of the PE (e.g., BIG, LTL, ACC, MEM, etc.)
        @param name: Name of the current processing element
        @param ID: ID of the current processing element
        @param cluster_ID: ID of the cluster to which this PE belongs
        @param capacity: Number tasks that a resource can run simultaneously
        '''
        
        self.env = env
        self.type = type
        self.name = name
        self.ID = ID
        self.capacity = capacity                                                # Current capacity of the PE (depends on the number of active cores)
        self.total_capacity = capacity                                          # Total capacity of the PE
        self.cluster_ID = cluster_ID
        self.throughput = throughput  
        self.power = power  
        self.total_energy = 0.0

        self.enabled = True                                                     # Indicate if the PE is ON
        self.utilization = 0                                                    # Describes how much one PE is utilized
        self.utilization_list = []                                              # List containing the PE utilization for each sample inside a snippet
        self.current_power_active_core = 0                                      # Indicate the current power for the active cores (dynamic + static)
        self.current_leakage_core = 0                                           # Indicate the current leakage power
        self.snippet_energy = 0                                                 # Indicate the energy consumption of the current snippet
                                                         # Indicate the total energy consumed by the given PE

        self.Cdyn_alpha = 0                                                     # Variable that stores the dynamic capacitance * switching activity for each PE

        self.queue = []                                                         # List of currently running task on a PE
        self.available_time = 0                                                 # Estimated available time of the PE
        self.available_time_list = [0]*self.capacity                            # Estimated available time for each core os the PE
        self.idle = True                                                        # The variable indicates whether the PE is active or not
        self.blocking = 0                                                       # Duration that a PE is busy when some other tasks are ready to execute 
        self.active = 0                                                         # Total active time for a PE while executing a workload
        
        self.info = []                                                          # List to record all the events happened on a PE
        
        
        
        

        
        self.process = simpy.Resource(env, capacity=self.capacity)

        if (common.DEBUG_CONFIG):
            print('[D] Constructed PE-%d with name %s' %(ID,name))

    # Start the "run" process for this PE
    def run(self, sim_manager, task, resource, DVFS_module=None):
        '''!
        Run this PE to execute a given task.
        This version removes all DTPM (DVFS, power, temperature) logic.
        '''
    
        try:
            with self.process.request() as req:
                yield req
    
                self.idle = False
                common.TaskQueues.running.list.append(task)
                task.start_time = self.env.now
    
                if ((task.head == True) and (self.env.now >= common.warmup_period)):
                    common.results.injected_jobs += 1
                    if common.DEBUG_JOB:
                        print('[D] Time %d: Total injected jobs becomes: %d' %
                              (self.env.now, common.results.injected_jobs))
                    if common.simulation_mode == 'validation':
                        common.Validation.injected_jobs.append(task.jobID)
    
                if common.DEBUG_JOB:
                    print('[D] Time %d: Task %s execution is started by PE-%d %s'
                          % (self.env.now, task.ID, self.ID, self.name))
    
                #  DTPM REMOVED — use fixed execution time
                exec_time = resource.performance[resource.supported_functionalities.index(task.name)]
                # print(f" ----------------exec_time{exec_time}")
                yield self.env.timeout(exec_time)
                task.finish_time = self.env.now
                task_time = task.finish_time - task.start_time
                # ==== Compute Energy (Fixed Power × Execution Time) ====

                energy_task = task_time * self.power    # time × power
                self.total_energy += energy_task        # accumulate per-PE
                common.results.energy_consumption += energy_task  # accumulate global
                # print(f" ----------------energy_task{energy_task}")

                if common.INFO_SIM:
                    print(f"[E] Time {self.env.now}: PE-{self.ID} ({self.name}) ran {task.name} "
                        f"for {task_time:.2f} us @ {self.power:.2f} W → Energy {energy_task:.4f} uJ")

    
                self.idle = True
                if task.finish_time > common.warmup_period:
                    if task.start_time <= common.warmup_period:
                        self.active += (task.finish_time - common.warmup_period)
                    else:
                        self.active += task_time
    
                if (task.tail):
                    common.results.job_counter -= 1
    
                    if common.simulation_mode == 'performance':
                        sim_manager.update_completed_queue()
    
                    if self.env.now >= common.warmup_period:
                        common.results.execution_time = self.env.now
                        common.results.completed_jobs += 1
    
                        if sim_manager.job_gen.generate_job and common.inject_jobs_ASAP:
                            sim_manager.job_gen.action.interrupt()
    
                        for completed in common.TaskQueues.completed.list:
                            if (completed.head and completed.jobID == task.jobID):
                                common.results.cumulative_exe_time += (self.env.now - completed.job_start)
    
                                if common.DEBUG_JOB:
                                    print('[D] Time %d: Job %d is completed' %
                                          (self.env.now, task.jobID + 1))
    
                    if common.simulation_mode == 'validation':
                        common.Validation.completed_jobs.append(task.jobID)
    
                # Energy: set to zero since DTPM is off
                DASH_Sim_utils.trace_tasks(task, self, task_time, energy_task)
    
                # if common.INFO_SIM:
                #     print('[I] Time %d: Task %s is finished by PE-%d %s with %.2f us and energy consumption %.2f J'
                #           % (self.env.now, task.ID, self.ID, self.name, round(task_time, 2), round(total_energy_task, 2)))
    
                
                # common.results.energy_consumption += total_energy_task
    
                sim_manager.update_ready_queue(task)
    
                if (task.tail and self.env.now >= common.warmup_period and
                        common.results.completed_jobs % common.snippet_size == 0):
    
                    for PE in sim_manager.PEs:
                        PE.snippet_energy = 0
    
                    common.snippet_start_time = self.env.now
                    common.snippet_initial_temp = copy.deepcopy(common.current_temperature_vector)
                    common.snippet_throttle = -1
    
                    for cluster in common.ClusterManager.cluster_list:
                        cluster.snippet_power_list = []
    
                    common.snippet_temp_list = []
                    common.snippet_ID_exec += 1
                    if common.job_list != []:
                        if common.snippet_ID_exec < common.max_num_jobs / common.snippet_size:
                            common.current_job_list = common.job_list[common.snippet_ID_exec]
    
                    if common.results.completed_jobs == common.max_num_jobs:
                        sim_manager.sim_done.succeed()
    
        except simpy.Interrupt:
            print('Expect an interrupt at %s' % (self.env.now))



# end class PE(object):
