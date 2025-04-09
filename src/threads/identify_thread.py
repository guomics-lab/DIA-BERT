import multiprocessing

from pubsub import pub

from src import common_logger
from src.common import constant
from src.common.obj import InputParam
from src.common.runtime_data_info import runtime_data
from src.identify_process_handler import IdentifyProcessHandler
from src.utils import msg_send_utils


class IdentifyThread(multiprocessing.Process):

    def __init__(self, input_param: InputParam, msg_qu: multiprocessing.Queue):
        multiprocessing.Process.__init__(self)
        self.input_param = input_param
        self.msg_qu = msg_qu


    def sub_msg(self, msg):
        if self.input_param.env == constant.env_win:
            self.msg_qu.put((msg, runtime_data))
        else:
            pass

    def run(self):
        current_logger, logger_file_path = common_logger.create_new_logger()
        self.input_param.logger_file_path = logger_file_path
        runtime_data.running_flag = True
        pub.subscribe(self.sub_msg, constant.msg_channel)
        msg_send_utils.send_msg(msg=
            'Start to identify, input_param: {}, logger path: {}'.format(self.input_param.__dict__, logger_file_path))
        idp = IdentifyProcessHandler(self.input_param, current_logger)
        try:
            idp.deal_process()
        except Exception as e:
            current_logger.exception('Identify exception')
            msg_send_utils.send_msg(msg='Identify exception: {}, detail info you can see log txt'.format(e))
        msg_send_utils.send_msg(status=constant.ProgressStepStatusEnum.ALL_END, msg='Finished')
