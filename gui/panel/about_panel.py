import wx

from gui import gui_config
from src.common import constant


class AboutInfoPanel(wx.Panel):

    def __init__(self, parent, id=wx.ID_ANY, pos=wx.DefaultPosition, size=wx.Size(400, 100), style=wx.TAB_TRAVERSAL,
                 name=wx.EmptyString):
        wx.Panel.__init__(self, parent, id=id, pos=pos, size=size, style=style, name=name)

        logo_img = wx.Image('./resource/logo/logo.png', wx.BITMAP_TYPE_ANY)
        logo_img.Rescale(256, 256)
        self.logo_img_bitmap = wx.StaticBitmap(self, wx.ID_ANY,
                                        wx.Bitmap(logo_img),
                                        wx.DefaultPosition, wx.DefaultSize, 0)
        self.logo_img_bitmap.SetWindowStyleFlag(wx.ALIGN_CENTER | wx.ALIGN_CENTER_VERTICAL)
        #
        blank_text = wx.StaticText(self, label='')
        blank_text.SetFont(gui_config.version_font)

        version_text = wx.StaticText(self, label=constant.VERSION)
        version_text.SetFont(gui_config.version_font)
        version_text.SetWindowStyleFlag(wx.ALIGN_CENTER | wx.ALIGN_CENTER_VERTICAL)

        self.about_text = wx.StaticText(self, label=gui_config.introduce_text)
        self.about_text.SetFont(gui_config.introduce_font)
        self.about_text.SetWindowStyleFlag(wx.ALIGN_CENTER | wx.ALIGN_CENTER_VERTICAL)
        # self.about_text.SetSize((-1, 600))

        about_text_sizer = wx.BoxSizer(wx.HORIZONTAL)
        # 在StaticText周围添加间距空白
        about_text_sizer.AddSpacer(100)  # 左侧间距
        about_text_sizer.Add(self.about_text, 1, wx.EXPAND)
        about_text_sizer.AddSpacer(100)  # 右侧间距

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.logo_img_bitmap, 0, wx.EXPAND | wx.ALL, 10)
        sizer.Add(about_text_sizer, 3, wx.EXPAND | wx.ALL, 10)
        sizer.Add(version_text, 0, wx.EXPAND | wx.ALL, 10)
        sizer.Add(blank_text, 0, wx.EXPAND | wx.ALL, 10)

        self.SetSizer(sizer)

    def __del__(self):
        pass
