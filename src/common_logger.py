import json
import logging
import logging.config as log_config
import os
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

cwd = os.getcwd()
log_dir = os.path.join(cwd, 'logs')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
log_config_path = os.path.join(cwd, 'config/log.config')
if os.path.exists(log_config_path):
    with open(log_config_path, 'rt') as f:
        config = json.load(f)
    log_config.dictConfig(config)
else:
    logging.basicConfig(level=logging.DEBUG)


def create_new_logger(log_dir='logs', log_file_name=None):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_format = logging.Formatter('DIA-BERT: %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    current_logger = logging.getLogger('DIA-BERT')
    if not log_file_name:
        log_file_name = 'DIA-BERT_{}.log'.format(time.time_ns())
    log_file_path = os.path.join(log_dir, log_file_name)
    fh = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    fh.setFormatter(log_format)
    fh.setLevel(logging.INFO)
    current_logger.addHandler(fh)
    return current_logger, log_file_path

