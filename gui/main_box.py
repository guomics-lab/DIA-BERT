import wx
import wx.grid
from pubsub import pub

from gui import gui_config
from gui.event_handler.run_panel_event_handler import RunPanelEventHandler
from gui.event_handler.run_panel_msg_sub_handler import RunPanelMsgSubHandler
from gui.panel.about_panel import AboutInfoPanel
from gui.panel.run_panel import RunInfoPanel
from src.common import constant
from src.common.runtime_data_info import runtime_data
from src.threads.msg_sub_thread import MsgSubThread


class MainListBook(wx.Listbook):

    def __init__(self, parent):
        wx.Listbook.__init__(self, parent, wx.ID_ANY)
        imagelist = wx.ImageList(64, 64)
        imagelist.Add(wx.Bitmap('./resource/icon/about.png', wx.BITMAP_TYPE_ANY))
        imagelist.Add(wx.Bitmap('./resource/icon/set.png', wx.BITMAP_TYPE_ANY))
        imagelist.Add(wx.Bitmap('./resource/icon/draw.png', wx.BITMAP_TYPE_ANY))
        # 将ImageList对象加入到m_listbook1里面，供后结菜单使用
        self.AssignImageList(imagelist)


class MainBox(wx.Frame):

    def __init__(self, ):
        # logger.info('start init frame')

        wx.Frame.__init__(self, None, title=constant.VERSION,
                          size=(1320, 860))

        self.SetIcon(wx.Icon('./resource/logo/logo.png', wx.BITMAP_TYPE_PNG))

        self.Centre(wx.BOTH)

        main_panel = wx.Panel(self)

        self.notebook = MainListBook(main_panel)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.notebook, 1, wx.ALL | wx.EXPAND, 5)
        main_panel.SetSizer(sizer)

        self.about_info_panel = AboutInfoPanel(self.notebook)
        self.run_info_panel = RunInfoPanel(self.notebook)

        self.notebook.AddPage(self.about_info_panel, 'About', select=True, imageId=0)
        self.notebook.AddPage(self.run_info_panel, 'Manual mode', imageId=1)

        self.rp_event_handler = RunPanelEventHandler(self.run_info_panel)

        self.Bind(wx.EVT_BUTTON, self.rp_event_handler.lib_btn_click, self.run_info_panel.config_panel.lib_btn)
        self.Bind(wx.EVT_BUTTON, self.rp_event_handler.mzml_choose_click, self.run_info_panel.input_panel.mzml_select_button)
        self.Bind(wx.EVT_BUTTON, self.rp_event_handler.clear_btn_click, self.run_info_panel.input_panel.clear_button)
        self.Bind(wx.EVT_BUTTON, self.rp_event_handler.output_dir_choose, self.run_info_panel.input_panel.output_path_choose_button)

        self.Bind(wx.EVT_BUTTON, self.rp_event_handler.run_btn_click, self.run_info_panel.run_control_panel.run_button)
        self.Bind(wx.EVT_BUTTON, self.rp_event_handler.stop_btn_click, self.run_info_panel.run_control_panel.stop_button)

        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.msg_sub_handler = RunPanelMsgSubHandler(self.run_info_panel)
        pub.subscribe(self.msg_sub_handler.sub_msg, constant.main_msg_channel)

        # 启动消息转发线程
        runtime_data.msg_sub_thread = MsgSubThread(constant.msg_queue)
        runtime_data.msg_sub_thread.start()

    def on_close(self, event):
        if runtime_data.identify_thread:
            runtime_data.identify_thread.terminate()
            runtime_data.identify_thread.join()
        if runtime_data.msg_sub_thread:
            constant.msg_queue.put((constant.QUEUE_END_FLAG, None))
            runtime_data.msg_sub_thread.run_flag = False
            runtime_data.msg_sub_thread.join()
        event.Skip()


