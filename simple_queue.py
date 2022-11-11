import numpy as np
SEED = 42

class Client:
    def __init__(self, type, arrival_time):
        self.type = type
        self.arrival_time = arrival_time


class Simulator:
    def __init__(self, arrival_factor, service_factor, verbose, endtime, seed):
        self.users = 0
        self.time = 0
        self.endtime = endtime
        self.verbose = verbose
        self.queue = list()
        self.fes = list()
        self.arrival_factor = arrival_factor
        self.service_factor = service_factor
        self.inter_arrival_generator = np.random.default_rng(seed=seed)
        self.serice_time_generator = np.random.default_rng(seed=seed)
        # scheduling the first arrival
        self.create_event(time=self.generate_inter_arrival_time(), action=self.arrival)

    def generate_service_time(self):
        service_time = self.serice_time_generator.exponential(1.0/self.service_factor)
        if (self.verbose == True):
            print("Time " + str(self.time) + ": scheduling job departure in " + str(service_time) + " time units")
        return service_time

    def generate_inter_arrival_time(self):
        inter_arrival = self.inter_arrival_generator.exponential(1.0/self.arrival_factor)
        if (self.verbose == True):
            print("Time " + str(self.time) + ": scheduling job arrival in " + str(inter_arrival) + " time units")
        return inter_arrival

    def create_event(self, time, action):
        self.fes.append({"time": time, "action": action})

    def arrival(self):
        # schedule the next arrival
        self.create_event(time=self.time+self.generate_inter_arrival_time(), action=self.arrival)
        # the client arrives in the queue
        self.users += 1
        self.queue.append(Client("type1", self.time))
        if (self.users == 1):
            # there wasn't any client in the queue, then we serve it immediately
            self.create_event(time=self.time+self.generate_service_time(), action=self.departure)
    
    def departure(self):
        client = self.queue.pop(0)
        # remove the client from the queue only at departure time
        self.users -= 1
        if (self.users > 0):
            # if there was another client in the queue, then serve it
            self.create_event(time=self.time+self.generate_service_time(), action=self.departure)

    def start(self):
        while (self.time < self.endtime):
            next_event = self.fes.pop(0)
            next_event["action"]()             # execute the next event
            self.time = next_event["time"]     # advance the simulator in time after the event is completed


def __main__():
    sim = Simulator(arrival_factor=2, service_factor=2, seed=SEED, endtime=10000, verbose=True)
    sim.start()


if __name__ == "__main__":
    __main__()
