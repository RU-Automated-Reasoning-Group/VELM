import copy 

class dataset:
    def __init__(self):
        self.states = []
        self.actions = []
        self.next_states = []
    
    def add_new_safe_data(self, new_data):
        for (state, action, next_state) in new_data:
            self.states.append(copy.deepcopy(state))
            self.actions.append(copy.deepcopy(action))
            self.next_states.append(copy.deepcopy(next_state))

    def add_new_data_DSO(self, replay_buffer):
        pass

    def add_new_data_operon(self, new_data):
        self.add_new_safe_data(new_data)

    def check_model_accuracy(self, model):
        pass

    def get_data_for_operon(self):
        buffer = []
        for s, a, ns in zip(self.states, self.actions, self.next_states):
            buffer.append(copy.deepcopy((s, a, ns)))
        return buffer
