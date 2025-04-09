
import multiprocessing
import threading

from pubsub import pub

from src.common import constant
from src.common.runtime_data_info import runtime_data
from src.common_logger import logger


class MsgSubThread(threading.Thread):

    def __init__(self, msg_queue: multiprocessing.Queue):
        threading.Thread.__init__(self)
        self.msg_queue = msg_queue
        self.run_flag = True

    def run(self):
        while self.run_flag:
            try:
                #
                (new_msg, runtime_data_info) = self.msg_queue.get()
                if new_msg == constant.QUEUE_END_FLAG:
                    break
                self.copy_runtime_data(runtime_data_info)
                pub.sendMessage(constant.main_msg_channel, msg=new_msg)
            except Exception:
                logger.exception('Share message exception')

    def copy_runtime_data(self, inner_runtime_data):
        runtime_data.mzml_deal_count = inner_runtime_data.mzml_deal_count
        runtime_data.start_timestamp = inner_runtime_data.start_timestamp
        runtime_data.current_mzml_index = inner_runtime_data.current_mzml_index
        runtime_data.current_is_success = inner_runtime_data.current_is_success
        runtime_data.current_identify_num = inner_runtime_data.current_identify_num
        runtime_data.current_identify_all_num = inner_runtime_data.current_identify_all_num
        runtime_data.running_flag = inner_runtime_data.running_flag

