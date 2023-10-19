# ETL super class for different api to inherit from
class ETL:
    def __init__(self, api_keys, start_day=None, end_day=None):
        self.api_keys = api_keys
        self.start_day = start_day
        self.end_day = end_day

    
    


