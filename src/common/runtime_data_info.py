
class RuntimeData(object):

    def __init__(self):
        self.mzml_list = []
        self.mzml_deal_count = 0

        self.identify_thread = None

        self.start_timestamp = None
        self.current_mzml_index = None
        self.current_is_success = True
        self.current_identify_num = None
        self.current_identify_all_num = None

        self.running_flag = True


runtime_data = RuntimeData()

