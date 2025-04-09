import json
import time

from pubsub import pub

from src.common import constant
from src.common import runtime_data_info
from src.common.obj import IdentifyMsg


def send_msg(mzml_name=None, mzml_index=None, step=None, status=None, msg=None, channel=None):
    #
    start_timestamp = runtime_data_info.runtime_data.start_timestamp
    if start_timestamp is None:
        pass
    else:
        if msg is None:
            pass
        else:
            msg = get_now_use_time(start_timestamp) + msg
    mzml_index = runtime_data_info.runtime_data.current_mzml_index
    info_msg = IdentifyMsg(mzml_name=mzml_name, mzml_index=mzml_index, step=step, status=status, msg=msg)
    if channel is None:
        channel = constant.msg_channel
    pub.sendMessage(channel, msg=json.dumps(info_msg.__dict__))


def get_now_use_time(start_timestamp):
    now_time = time.time()
    minutes, seconds = divmod(now_time - start_timestamp, 60)
    minutes = int(minutes)
    seconds = int(seconds)
    return '[{}:{}]'.format(minutes, str(seconds).zfill(2))