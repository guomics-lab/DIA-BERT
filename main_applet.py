import multiprocessing

import wx

from gui.main_box import MainBox

from src.common_logger import create_new_logger


def create_frame():
    app = wx.App()
    frame = MainBox()
    frame.Centre()
    frame.Show()

    app.MainLoop()


if __name__ == '__main__':
    current_logger, log_file_path = create_new_logger()
    try:
        multiprocessing.freeze_support()
        create_frame()
    except Exception:
        current_logger.exception('Start error')

