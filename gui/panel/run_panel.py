import wx

from src.utils.gpu_utils import get_top_free_device, get_usage_device
from gui import gui_config
from src import common_config
from src.common import constant

btn_width = 120

label_width = 100

text_width = 150

text_span = 2

common_font = gui_config.common_font
common_config_data = common_config.read_yml()


class RunInfoPanel(wx.Panel):

    def __init__(self, parent, id=wx.ID_ANY, pos=wx.DefaultPosition, size=wx.Size(400, 100), style=wx.TAB_TRAVERSAL,
                 name=wx.EmptyString):
        wx.Panel.__init__(self, parent, id=id, pos=pos, size=size, style=style, name=name)

        run_info_gb_sizer = wx.GridBagSizer(0, 0)
        run_info_gb_sizer.SetFlexibleDirection(wx.BOTH)
        run_info_gb_sizer.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)

        self.config_panel = ConfigPanel(self)
        self.input_panel = InputPanel(self)
        self.run_control_panel = RunControlPanel(self)
        self.run_info_display_panel = RunInfoDisplayPanel(self)
        self.log_panel = LogPanel(self)

        run_info_gb_sizer.Add(self.config_panel, wx.GBPosition(0, 0), wx.GBSpan(5, 7), wx.ALL, 5)
        run_info_gb_sizer.Add(self.input_panel, wx.GBPosition(5, 0), wx.GBSpan(2, 7), wx.ALL, 5)
        run_info_gb_sizer.Add(self.run_info_display_panel, wx.GBPosition(0, 7), wx.GBSpan(4, 4), wx.ALL | wx.EXPAND, 5)
        run_info_gb_sizer.Add(self.run_control_panel, wx.GBPosition(4, 7), wx.GBSpan(1, 4), wx.ALL, 5)
        run_info_gb_sizer.Add(self.log_panel, wx.GBPosition(5, 7), wx.GBSpan(4, 4), wx.ALL | wx.EXPAND, 5)

        run_info_gb_sizer.AddGrowableRow(5)
        run_info_gb_sizer.AddGrowableCol(9)

        self.SetSizer(run_info_gb_sizer)
        self.Layout()

    def __del__(self):
        pass


class ConfigPanel(wx.Panel):

    def __init__(self, parent, id=wx.ID_ANY, pos=wx.DefaultPosition, size=wx.Size(550, 500), style=wx.TAB_TRAVERSAL,
                 name=wx.EmptyString):
        wx.Panel.__init__(self, parent, id=id, pos=pos, size=size, style=style, name=name)
        # import torch
        # current_gpu_index = torch.cuda.current_device()
        # total_memory = torch.cuda.get_device_properties(current_gpu_index).total_memory / (1024 ** 3)  # 显存总量(GB)
        # used_memory = torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3)  # 已使用显存(GB)
        # free_memory = total_memory - used_memory  # 剩余显存(GB)
        # batch_size = int(free_memory * 12.5) # 按照剩余显存，更新step_size
        # step_size = int(free_memory * 500) # 按照剩余显存，更新step_size

        # 获取可用的GPU的数量，使用率小于50%的
        usage_device_list, min_free_memory = get_usage_device(common_config_data['gpu']['max_member_use_rate'])
        max_gpu_num = len(usage_device_list)

        batch_size = int(min_free_memory * 12.5) # 按照剩余显存，更新step_size
        step_size = int(min_free_memory * 500) # 按照剩余显存，更新step_size


        input_sb = wx.StaticBox(self, wx.ID_ANY, u"Configuration")
        input_sb.SetFont(common_font)
        input_sb_sizer = wx.StaticBoxSizer(input_sb, wx.VERTICAL)

        input_gb_sizer = wx.GridBagSizer(0, 0)
        input_gb_sizer.SetFlexibleDirection(wx.BOTH)
        input_gb_sizer.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)

        self.lib_btn = wx.Button(input_sb_sizer.GetStaticBox(), wx.ID_ANY, u"Spectral", wx.DefaultPosition,
                                 wx.Size(label_width, -1), 0)
        self.lib_btn.SetFont(common_font)

        input_gb_sizer.Add(self.lib_btn, wx.GBPosition(0, 0), wx.GBSpan(1, 2), wx.ALL, 5)

        lib_tooltip = wx.ToolTip("Select the spectral library")
        # self.lib_btn.SetToolTip(lib_tooltip)

        self.lib_path_text = wx.TextCtrl(input_sb_sizer.GetStaticBox(), wx.ID_ANY, r'',
                                         wx.DefaultPosition, wx.Size(250, -1), 0)
        self.lib_path_text.SetFont(common_font)
        input_gb_sizer.Add(self.lib_path_text, wx.GBPosition(0, 2), wx.GBSpan(1, 7), wx.ALL | wx.EXPAND, 5)

        device_label = wx.StaticText(self, wx.ID_ANY, u"Device", wx.DefaultPosition, (label_width, -1), wx.ALIGN_RIGHT)
        device_label.Wrap(-1)
        input_gb_sizer.Add(device_label, wx.GBPosition(1, 0), wx.GBSpan(1, 2), wx.ALL | wx.ALIGN_RIGHT, 5)

        self.gpu_btn = wx.RadioButton(input_sb_sizer.GetStaticBox(), wx.ID_ANY, u"GPU", wx.DefaultPosition,
                                      (text_width, -1), wx.RB_GROUP)
        input_gb_sizer.Add(self.gpu_btn, wx.GBPosition(1, 2), wx.GBSpan(1, 2), wx.ALL, 5)

        thread_num_label = wx.StaticText(self, wx.ID_ANY, u"Run threads", wx.DefaultPosition, (label_width, -1),
                                         wx.ALIGN_RIGHT)
        thread_num_label.Wrap(-1)
        thread_num_tooltip = wx.ToolTip("Thread num of run")
        # thread_num_label.SetToolTip(thread_num_tooltip)
        input_gb_sizer.Add(thread_num_label, wx.GBPosition(1, 4), wx.GBSpan(1, 2),
                           wx.ALL | wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)

        self.thread_num_spin_ctrl = wx.SpinCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, (text_width, -1),
                                                wx.SP_ARROW_KEYS, 0, 100,
                                                int(common_config_data['identify']['default_thread']))
        input_gb_sizer.Add(self.thread_num_spin_ctrl, wx.GBPosition(1, 6), wx.GBSpan(1, 2), wx.ALL, 5)


        gpu_num_label = wx.StaticText(self, wx.ID_ANY, u"Use GPU num", wx.DefaultPosition, (label_width, -1),
                                         wx.ALIGN_RIGHT)
        gpu_num_label.Wrap(-1)
        input_gb_sizer.Add(gpu_num_label, wx.GBPosition(2, 0), wx.GBSpan(1, 2),
                           wx.ALL | wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)

        self.gpu_num_spin_ctrl = wx.SpinCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, (text_width, -1),
                                                wx.SP_ARROW_KEYS, 0, int(max_gpu_num),
                                                int(max_gpu_num))
        input_gb_sizer.Add(self.gpu_num_spin_ctrl, wx.GBPosition(2, 2), wx.GBSpan(1, 2), wx.ALL, 5)

        spectral_config_start_row = 3
        spectral_config_label = wx.StaticText(self, wx.ID_ANY, u"Spectral config", wx.DefaultPosition,
                                              (label_width, -1), wx.ALIGN_RIGHT)
        spectral_config_label.Wrap(-1)
        input_gb_sizer.Add(spectral_config_label, wx.GBPosition(spectral_config_start_row, 0), wx.GBSpan(1, 2),
                           wx.ALL | wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        line1 = wx.StaticLine(self, wx.ID_ANY, wx.DefaultPosition, (1, 1), wx.LI_VERTICAL)
        input_gb_sizer.Add(line1, wx.GBPosition(spectral_config_start_row, 2), wx.GBSpan(1, 6), wx.ALL | wx.EXPAND, 5)

        decoy_method_label = wx.StaticText(self, wx.ID_ANY, u"Decoy method", wx.DefaultPosition, (label_width, -1),
                                           wx.ALIGN_RIGHT)
        decoy_method_label.Wrap(-1)
        decoy_method_tooltip = wx.ToolTip("Decoy method")
        # decoy_method_label.SetToolTip(decoy_method_tooltip)
        input_gb_sizer.Add(decoy_method_label, wx.GBPosition(spectral_config_start_row + 1, 0), wx.GBSpan(1, 2),
                           wx.ALL | wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        self.decoy_method_choice = wx.Choice(self, wx.ID_ANY, wx.DefaultPosition, (text_width, -1),
                                             constant.decoy_method_list, 0)
        input_gb_sizer.Add(self.decoy_method_choice, wx.GBPosition(spectral_config_start_row + 1, 2), wx.GBSpan(1, 2),
                           wx.ALL, 5)
        self.decoy_method_choice.Select(0)

        rt_norm_label = wx.StaticText(self, wx.ID_ANY, u"RT normalization", wx.DefaultPosition, (label_width, -1),
                                      wx.ALIGN_RIGHT)
        rt_norm_label.Wrap(-1)
        rt_norm_tooltip = wx.ToolTip("RT normalization num")
        # rt_norm_label.SetToolTip(rt_norm_tooltip)
        input_gb_sizer.Add(rt_norm_label, wx.GBPosition(spectral_config_start_row + 1, 4), wx.GBSpan(1, 2),
                           wx.ALL | wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        self.rt_norm_spin_ctrl = wx.SpinCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, (text_width, -1),
                                             wx.SP_ARROW_KEYS, 0, 100000,
                                             int(common_config_data['identify']['default_rt_normalization']))
        input_gb_sizer.Add(self.rt_norm_spin_ctrl, wx.GBPosition(spectral_config_start_row + 1, 6), wx.GBSpan(1, 2),
                           wx.ALL, 5)

        mz_rt_unit_label = wx.StaticText(self, wx.ID_ANY, u"m/z unit", wx.DefaultPosition, (label_width, -1),
                                         wx.ALIGN_RIGHT)
        mz_rt_unit_label.Wrap(-1)
        mz_rt_unit_tooltip = wx.ToolTip("MZ RT unit")
        # mz_rt_unit_label.SetToolTip(mz_rt_unit_tooltip)
        input_gb_sizer.Add(mz_rt_unit_label, wx.GBPosition(spectral_config_start_row + 2, 0), wx.GBSpan(1, 2),
                           wx.ALL | wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        self.mz_rt_unit_choice = wx.Choice(self, wx.ID_ANY, wx.DefaultPosition, (text_width, -1),
                                           constant.mz_rt_unit_list, 0)
        input_gb_sizer.Add(self.mz_rt_unit_choice, wx.GBPosition(spectral_config_start_row + 2, 2), wx.GBSpan(1, 2),
                           wx.ALL, 5)
        self.mz_rt_unit_choice.Select(0)

        fitting_rt_batch_size_label = wx.StaticText(self, wx.ID_ANY, u"RT peak size", wx.DefaultPosition,
                                                    (label_width, -1), wx.ALIGN_RIGHT)
        fitting_rt_batch_size_label.Wrap(-1)
        rt_norm_tooltip = wx.ToolTip("RT normalization each peak size")
        # fitting_rt_batch_size_label.SetToolTip(rt_norm_tooltip)
        input_gb_sizer.Add(fitting_rt_batch_size_label, wx.GBPosition(spectral_config_start_row + 2, 4),
                           wx.GBSpan(1, 2), wx.ALL | wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        self.fitting_rt_batch_size_spin_ctrl = wx.SpinCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition,
                                                           (text_width, -1), wx.SP_ARROW_KEYS, 0, 5000,
                                                           int(common_config_data['identify']['default_rt_peak_size']))
        input_gb_sizer.Add(self.fitting_rt_batch_size_spin_ctrl, wx.GBPosition(spectral_config_start_row + 2, 6),
                           wx.GBSpan(1, 2), wx.ALL, 5)

        raw_info_start_row = 6
        raw_info_label = wx.StaticText(self, wx.ID_ANY, u"Raw info config", wx.DefaultPosition, (label_width, -1),
                                       wx.ALIGN_RIGHT)
        raw_info_label.Wrap(-1)
        input_gb_sizer.Add(raw_info_label, wx.GBPosition(raw_info_start_row, 0), wx.GBSpan(1, 2),
                           wx.ALL | wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        line2 = wx.StaticLine(self, wx.ID_ANY, wx.DefaultPosition, (1, 1), wx.LI_VERTICAL)
        input_gb_sizer.Add(line2, wx.GBPosition(raw_info_start_row, 2), wx.GBSpan(1, 6), wx.ALL | wx.EXPAND, 5)

        rt_unit_label = wx.StaticText(self, wx.ID_ANY, u"RT unit", wx.DefaultPosition, (label_width, -1),
                                      wx.ALIGN_RIGHT)
        rt_unit_label.Wrap(-1)
        rt_unit_tooltip = wx.ToolTip("Raw RT unit")
        # rt_unit_label.SetToolTip(rt_unit_tooltip)
        input_gb_sizer.Add(rt_unit_label, wx.GBPosition(raw_info_start_row + 1, 0), wx.GBSpan(1, 2),
                           wx.ALL | wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        self.rt_unit_choice = wx.Choice(self, wx.ID_ANY, wx.DefaultPosition, (text_width, -1),
                                        constant.raw_rt_unit_list, 0)
        input_gb_sizer.Add(self.rt_unit_choice, wx.GBPosition(raw_info_start_row + 1, 2), wx.GBSpan(1, 2), wx.ALL, 5)
        self.rt_unit_choice.Select(0)

        instrument_label = wx.StaticText(self, wx.ID_ANY, u"Instrument", wx.DefaultPosition, (label_width, -1),
                                         wx.ALIGN_RIGHT)
        instrument_label.Wrap(-1)
        instrument_tooltip = wx.ToolTip("Instrument of the raw")
        # instrument_label.SetToolTip(instrument_tooltip)
        input_gb_sizer.Add(instrument_label, wx.GBPosition(raw_info_start_row + 1, 4), wx.GBSpan(1, 2),
                           wx.ALL | wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        self.instrument_choice = wx.Choice(self, wx.ID_ANY, wx.DefaultPosition, (text_width, -1),
                                           constant.instrument_list, 0)
        input_gb_sizer.Add(self.instrument_choice, wx.GBPosition(raw_info_start_row + 1, 6), wx.GBSpan(1, 2), wx.ALL, 5)
        self.instrument_choice.Select(0)

        identify_config_start_raw = 8
        identify_config_label = wx.StaticText(self, wx.ID_ANY, u"Identify config", wx.DefaultPosition,
                                              (label_width, -1), wx.ALIGN_RIGHT)
        identify_config_label.Wrap(-1)
        input_gb_sizer.Add(identify_config_label, wx.GBPosition(identify_config_start_raw, 0), wx.GBSpan(1, 2),
                           wx.ALL | wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        line3 = wx.StaticLine(self, wx.ID_ANY, wx.DefaultPosition, (1, 1), wx.LI_VERTICAL)
        input_gb_sizer.Add(line3, wx.GBPosition(identify_config_start_raw, 2), wx.GBSpan(1, 6), wx.ALL | wx.EXPAND, 5)

        batch_size_label = wx.StaticText(self, wx.ID_ANY, u"Batch size", wx.DefaultPosition, (label_width, -1),
                                         wx.ALIGN_RIGHT)
        batch_size_label.Wrap(-1)
        batch_size_tooltip = wx.ToolTip("Each deal precursor num")
        # batch_size_label.SetToolTip(batch_size_tooltip)
        input_gb_sizer.Add(batch_size_label, wx.GBPosition(identify_config_start_raw + 1, 0), wx.GBSpan(1, 2),
                           wx.ALL | wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)

        # self.batch_size_spin_ctrl = wx.SpinCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, (text_width, -1),
        #                                         wx.SP_ARROW_KEYS, 0, 5000,
        #                                         int(common_config_data['identify']['default_batch_size']))
        self.batch_size_spin_ctrl = wx.SpinCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, (text_width, -1),
                                                wx.SP_ARROW_KEYS, 0, 5000, batch_size)

        input_gb_sizer.Add(self.batch_size_spin_ctrl, wx.GBPosition(identify_config_start_raw + 1, 2), wx.GBSpan(1, 2),
                           wx.ALL, 5)

        batch_score_size_label = wx.StaticText(self, wx.ID_ANY, u"Batch score size", wx.DefaultPosition,
                                               (label_width, -1), wx.ALIGN_RIGHT)
        batch_score_size_label.Wrap(-1)
        batch_size_tooltip = wx.ToolTip("Each deal score precursor num")
        # batch_score_size_label.SetToolTip(batch_size_tooltip)
        input_gb_sizer.Add(batch_score_size_label, wx.GBPosition(identify_config_start_raw + 1, 4), wx.GBSpan(1, 2),
                           wx.ALL | wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)

        # self.batch_score_size_spin_ctrl = wx.SpinCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition,
        #                                               (text_width, -1), wx.SP_ARROW_KEYS, 0, 50000,
        #                                               int(common_config_data['identify']['default_batch_score_size']))
        self.batch_score_size_spin_ctrl = wx.SpinCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition,
                                                      (text_width, -1), wx.SP_ARROW_KEYS, 0, 50000, step_size)
        input_gb_sizer.Add(self.batch_score_size_spin_ctrl, wx.GBPosition(identify_config_start_raw + 1, 6),
                           wx.GBSpan(1, 2), wx.ALL, 5)

        finetune_config_start_row = 10
        finetune_config_label = wx.StaticText(self, wx.ID_ANY, u"Finetune config", wx.DefaultPosition,
                                              (label_width, -1), wx.ALIGN_RIGHT)
        finetune_config_label.Wrap(-1)
        input_gb_sizer.Add(finetune_config_label, wx.GBPosition(finetune_config_start_row, 0), wx.GBSpan(1, 2),
                           wx.ALL | wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        line3 = wx.StaticLine(self, wx.ID_ANY, wx.DefaultPosition, (1, 1), wx.LI_VERTICAL)
        input_gb_sizer.Add(line3, wx.GBPosition(finetune_config_start_row, 2), wx.GBSpan(1, 6), wx.ALL | wx.EXPAND, 5)

        finetune_score_label = wx.StaticText(self, wx.ID_ANY, u"Finetune score", wx.DefaultPosition, (label_width, -1),
                                             wx.ALIGN_RIGHT)
        finetune_score_label.Wrap(-1)
        finetune_score_tooltip = wx.ToolTip("Finetune min score")
        # finetune_score_label.SetToolTip(finetune_score_tooltip)
        input_gb_sizer.Add(finetune_score_label, wx.GBPosition(finetune_config_start_row + 1, 0), wx.GBSpan(1, 2),
                           wx.ALL | wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)

        self.finetune_score_spin_ctrl = wx.SpinCtrlDouble(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition,
                                                          (text_width, -1), wx.SP_ARROW_KEYS, 0, 1, float(
                common_config_data['identify']['default_finetune_score']), 0.1)
        input_gb_sizer.Add(self.finetune_score_spin_ctrl, wx.GBPosition(finetune_config_start_row + 1, 2),
                           wx.GBSpan(1, 2), wx.ALL, 5)

        train_epochs_label = wx.StaticText(self, wx.ID_ANY, u"Train epochs", wx.DefaultPosition, (label_width, -1),
                                           wx.ALIGN_RIGHT)
        train_epochs_label.Wrap(-1)
        train_epochs_label_tooltip = wx.ToolTip("Finetune train epochs")
        # train_epochs_label.SetToolTip(train_epochs_label_tooltip)
        input_gb_sizer.Add(train_epochs_label, wx.GBPosition(finetune_config_start_row + 1, 4), wx.GBSpan(1, 2),
                           wx.ALL | wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)

        self.train_epochs_spin_ctrl = wx.SpinCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, (text_width, -1),
                                                  wx.SP_ARROW_KEYS, 0, 50,
                                                  int(common_config_data['identify']['default_train_epochs']))
        input_gb_sizer.Add(self.train_epochs_spin_ctrl, wx.GBPosition(finetune_config_start_row + 1, 6),
                           wx.GBSpan(1, 2), wx.ALL, 5)

        algorithm_config_start_row = 12
        algorithm_config_label = wx.StaticText(self, wx.ID_ANY, u"Algorithm  config", wx.DefaultPosition,
                                               (label_width, -1), wx.ALIGN_RIGHT)
        algorithm_config_label.Wrap(-1)
        input_gb_sizer.Add(algorithm_config_label, wx.GBPosition(algorithm_config_start_row, 0), wx.GBSpan(1, 2),
                           wx.ALL | wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        line3 = wx.StaticLine(self, wx.ID_ANY, wx.DefaultPosition, (1, 1), wx.LI_VERTICAL)
        input_gb_sizer.Add(line3, wx.GBPosition(algorithm_config_start_row, 2), wx.GBSpan(1, 6), wx.ALL | wx.EXPAND, 5)

        # quantification_label = wx.StaticText(self, wx.ID_ANY, u"Quantification", wx.DefaultPosition, (label_width, -1),
        #                                      wx.ALIGN_RIGHT)
        # quantification_label.Wrap(-1)
        # input_gb_sizer.Add(quantification_label, wx.GBPosition(algorithm_config_start_row + 1, 0), wx.GBSpan(1, 2),
        #                    wx.ALL | wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        #
        # self.quantification_open_check_box = wx.CheckBox(self, wx.ID_ANY, u"Cross-run", wx.DefaultPosition, (text_width, -1), 0)
        # input_gb_sizer.Add(self.quantification_open_check_box, wx.GBPosition(algorithm_config_start_row + 1, 2),
        #                    wx.GBSpan(1, 2), wx.ALL, 5)

        protein_infer_label = wx.StaticText(self, wx.ID_ANY, u"Protein inference", wx.DefaultPosition, (label_width, -1),
                                            wx.ALIGN_RIGHT)
        protein_infer_label.Wrap(-1)
        input_gb_sizer.Add(protein_infer_label, wx.GBPosition(algorithm_config_start_row + 1, 0), wx.GBSpan(1, 2),
                           wx.ALL | wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)

        self.protein_infer_choice = wx.Choice(self, wx.ID_ANY, wx.DefaultPosition, (text_width, -1),
                                              constant.protein_infer_key_list, 0)
        input_gb_sizer.Add(self.protein_infer_choice, wx.GBPosition(algorithm_config_start_row + 1, 2), wx.GBSpan(1, 2),
                           wx.ALL, 5)
        self.protein_infer_choice.Select(0)

        run_step_label = wx.StaticText(self, wx.ID_ANY, u"Step", wx.DefaultPosition, (label_width, -1),
                                             wx.ALIGN_RIGHT)
        run_step_label.Wrap(-1)
        input_gb_sizer.Add(run_step_label, wx.GBPosition(algorithm_config_start_row + 2, 0), wx.GBSpan(1, 2),
                           wx.ALL | wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)

        self.lib_decoy_check_box = wx.CheckBox(self, wx.ID_ANY, u"Lib decoy", wx.DefaultPosition,
                                                         (text_width/ 2, -1), 0)
        input_gb_sizer.Add(self.lib_decoy_check_box, wx.GBPosition(algorithm_config_start_row + 2, 2),
                           wx.GBSpan(1, 1), wx.ALL, 5)
        self.lib_decoy_check_box.SetValue(True)
        self.lib_decoy_check_box.Enable(False)

        self.identify_check_box = wx.CheckBox(self, wx.ID_ANY, u"Identify", wx.DefaultPosition,
                                                         (text_width / 2, -1), 0)
        input_gb_sizer.Add(self.identify_check_box, wx.GBPosition(algorithm_config_start_row + 2, 3),
                           wx.GBSpan(1, 1), wx.ALL, 5)
        self.identify_check_box.SetValue(True)

        self.cross_run_check_box = wx.CheckBox(self, wx.ID_ANY, u"Cross-run", wx.DefaultPosition,
                                                         (text_width / 2, -1), 0)
        input_gb_sizer.Add(self.cross_run_check_box, wx.GBPosition(algorithm_config_start_row + 2, 4),
                           wx.GBSpan(1, 1), wx.ALL, 5)

        input_sb_sizer.Add(input_gb_sizer, 1, wx.EXPAND, 5)

        input_gb_sizer.AddGrowableCol(6)

        self.SetSizer(input_sb_sizer)
        self.Layout()

    def __del__(self):
        pass


class InputPanel(wx.Panel):

    def __init__(self, parent, id=wx.ID_ANY, pos=wx.DefaultPosition, size=wx.Size(550, 300), style=wx.TAB_TRAVERSAL,
                 name=wx.EmptyString):
        wx.Panel.__init__(self, parent, id=id, pos=pos, size=size, style=style, name=name)

        input_sb = wx.StaticBox(self, wx.ID_ANY, u"Input")
        input_sb.SetFont(common_font)
        input_sb_sizer = wx.StaticBoxSizer(input_sb, wx.VERTICAL)

        input_gb_sizer = wx.GridBagSizer(0, 0)
        input_gb_sizer.SetFlexibleDirection(wx.BOTH)
        input_gb_sizer.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)

        self.mzml_select_button = wx.Button(input_sb_sizer.GetStaticBox(), wx.ID_ANY, u".mzML", wx.DefaultPosition,
                                            wx.Size(btn_width, -1), 0)
        self.mzml_select_button.SetFont(common_font)

        input_gb_sizer.Add(self.mzml_select_button, wx.GBPosition(0, 0), wx.GBSpan(1, 2), wx.ALL, 5)

        self.clear_button = wx.Button(input_sb_sizer.GetStaticBox(), wx.ID_ANY, u"Clear", wx.DefaultPosition,
                                      wx.Size(btn_width, -1), 0)
        self.clear_button.SetFont(common_font)

        input_gb_sizer.Add(self.clear_button, wx.GBPosition(0, 4), wx.GBSpan(1, 2), wx.ALL, 5)

        # self.mzml_file_path_text = wx.TextCtrl(input_sb_sizer.GetStaticBox(), wx.ID_ANY, wx.EmptyString,
        #                                       wx.DefaultPosition, wx.Size(380, 160), wx.TE_MULTILINE | wx.TE_RICH2)
        # self.mzml_file_path_text.SetFont(common_font)

        self.mzml_data_list = wx.ListCtrl(self, wx.ID_ANY, style=wx.LC_REPORT, size=(380, 160))
        self.mzml_data_list.InsertColumn(0, "File", format=wx.LIST_FORMAT_CENTRE, width=380)

        input_gb_sizer.Add(self.mzml_data_list, wx.GBPosition(1, 0), wx.GBSpan(5, 7), wx.ALL | wx.EXPAND, 5)

        self.output_path_choose_button = wx.Button(input_sb_sizer.GetStaticBox(), wx.ID_ANY, u"Output dir",
                                                   wx.DefaultPosition, wx.Size(btn_width, -1), 0)
        self.output_path_choose_button.SetFont(common_font)

        input_gb_sizer.Add(self.output_path_choose_button, wx.GBPosition(7, 0), wx.GBSpan(1, 2),
                           wx.ALIGN_RIGHT | wx.ALL, 5)

        self.file_output_dir_text = wx.TextCtrl(input_sb_sizer.GetStaticBox(), wx.ID_ANY, r'',
                                                wx.DefaultPosition, wx.Size(250, -1), 0)
        self.file_output_dir_text.SetFont(common_font)

        input_gb_sizer.Add(self.file_output_dir_text, wx.GBPosition(7, 2), wx.GBSpan(1, 6), wx.ALL | wx.EXPAND, 5)

        input_gb_sizer.AddGrowableCol(6)

        input_sb_sizer.Add(input_gb_sizer, 1, wx.EXPAND, 5)

        self.SetSizer(input_sb_sizer)
        self.Layout()

    def __del__(self):
        pass


class RunControlPanel(wx.Panel):

    def __init__(self, parent, id=wx.ID_ANY, pos=wx.DefaultPosition, size=wx.Size(400, 50), style=wx.TAB_TRAVERSAL,
                 name=wx.EmptyString):
        wx.Panel.__init__(self, parent, id=id, pos=pos, size=size, style=style, name=name)

        run_info_gb_sizer = wx.GridBagSizer(0, 0)
        run_info_gb_sizer.SetFlexibleDirection(wx.BOTH)
        run_info_gb_sizer.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)

        self.run_button = wx.Button(self, wx.ID_ANY, u"Run", wx.DefaultPosition, wx.DefaultSize, 0)
        self.run_button.SetFont(common_font)

        run_info_gb_sizer.Add(self.run_button, wx.GBPosition(1, 4), wx.GBSpan(1, 2), wx.ALL, 5)

        self.run_status_button = wx.Button(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size(25, -1),
                                           wx.BU_NOTEXT)
        run_info_gb_sizer.Add(self.run_status_button, wx.GBPosition(1, 7), wx.GBSpan(1, 1), wx.ALL, 5)

        self.run_status_label = wx.StaticText(self, wx.ID_ANY, u"Finished", wx.DefaultPosition, wx.DefaultSize, 0)
        self.run_status_label.SetFont(common_font)

        self.run_status_label.Wrap(-1)

        run_info_gb_sizer.Add(self.run_status_label, wx.GBPosition(1, 8), wx.GBSpan(1, 1), wx.ALIGN_CENTER | wx.ALL, 5)

        self.stop_button = wx.Button(self, wx.ID_ANY, u"Stop", wx.DefaultPosition, wx.DefaultSize, 0)
        self.stop_button.SetFont(common_font)

        run_info_gb_sizer.Add(self.stop_button, wx.GBPosition(1, 10), wx.GBSpan(1, 2), wx.ALL, 5)

        self.SetSizer(run_info_gb_sizer)
        self.Layout()

    def __del__(self):
        pass


class LogPanel(wx.Panel):
    def __init__(self, parent, pos=wx.DefaultPosition, style=wx.TAB_TRAVERSAL,
                 name=wx.EmptyString):
        wx.Panel.__init__(self, parent, pos=pos, style=style, name=name)
        log_sb = wx.StaticBox(self, wx.ID_ANY, u"Log")
        log_sb.SetFont(common_font)
        log_info_box_sizer = wx.StaticBoxSizer(log_sb, wx.VERTICAL)

        log_info_sizer = wx.GridBagSizer(0, 0)
        log_info_sizer.SetFlexibleDirection(wx.BOTH)
        log_info_sizer.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)

        self.log_text = wx.TextCtrl(log_info_box_sizer.GetStaticBox(), wx.ID_ANY, wx.EmptyString,
                                    wx.DefaultPosition, wx.DefaultSize,
                                    style=wx.TE_READONLY | wx.TE_MULTILINE | wx.TE_RICH2)
        # self.log_text.Enable(False)
        self.log_text.SetFont(common_font)

        log_info_sizer.Add(self.log_text, wx.GBPosition(0, 0), wx.GBSpan(1, 1), wx.ALL | wx.EXPAND, 5)

        log_info_box_sizer.Add(log_info_sizer, 1, wx.EXPAND, 5)

        log_info_sizer.AddGrowableCol(0)
        log_info_sizer.AddGrowableRow(0)

        self.SetSizer(log_info_box_sizer)
        self.Layout()

    def __del__(self):
        pass


class RunInfoDisplayPanel(wx.Panel):

    def __init__(self, parent, id=wx.ID_ANY, pos=wx.DefaultPosition, size=wx.Size(400, 300), style=wx.TAB_TRAVERSAL,
                 name=wx.EmptyString):
        wx.Panel.__init__(self, parent, id=id, pos=pos, size=size, style=style, name=name)

        ri_sb = wx.StaticBox(self, wx.ID_ANY, u"Run progress")
        ri_sb.SetFont(common_font)
        output_sb_sizer = wx.StaticBoxSizer(ri_sb, wx.VERTICAL)

        output_gb_sizer = wx.GridBagSizer(0, 0)
        output_gb_sizer.SetFlexibleDirection(wx.BOTH)
        output_gb_sizer.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)

        self.progress_data_list = wx.ListCtrl(self, wx.ID_ANY, style=wx.LC_REPORT, size=(580, 160))
        self.progress_data_list.InsertColumn(0, "File", wx.LIST_FORMAT_CENTRE, width=160)
        self.progress_data_list.InsertColumn(1, "Parse mzML", wx.LIST_FORMAT_CENTRE, width=200)
        self.progress_data_list.InsertColumn(2, "RT normalization", wx.LIST_FORMAT_CENTRE, width=200)
        self.progress_data_list.InsertColumn(3, "Screen", wx.LIST_FORMAT_CENTRE, width=200)
        self.progress_data_list.InsertColumn(4, "Prepare data", wx.LIST_FORMAT_CENTRE, width=160)
        self.progress_data_list.InsertColumn(5, "Finetune train", wx.LIST_FORMAT_CENTRE, width=160)
        self.progress_data_list.InsertColumn(6, "Finetune eval", wx.LIST_FORMAT_CENTRE, width=160)
        self.progress_data_list.InsertColumn(7, "Quant", wx.LIST_FORMAT_CENTRE, width=110)

        output_gb_sizer.Add(self.progress_data_list, wx.GBPosition(0, 0), wx.GBSpan(1, 18), wx.ALL | wx.EXPAND, 5)

        self.all_progress_label = wx.StaticText(output_sb_sizer.GetStaticBox(), wx.ID_ANY, u"Progress",
                                                wx.DefaultPosition, wx.Size(-1, -1), 0)
        self.all_progress_label.SetFont(common_font)
        self.all_progress_label.Wrap(-1)
        output_gb_sizer.Add(self.all_progress_label, wx.GBPosition(3, 1), wx.GBSpan(1, 2), wx.ALIGN_RIGHT | wx.ALL, 5)

        self.all_progress_gauge = wx.Gauge(output_sb_sizer.GetStaticBox(), wx.ID_ANY, 100, wx.DefaultPosition,
                                           wx.DefaultSize,
                                           wx.GA_HORIZONTAL)
        self.all_progress_gauge.SetValue(0)
        output_gb_sizer.Add(self.all_progress_gauge, wx.GBPosition(3, 3), wx.GBSpan(1, 14), wx.ALL | wx.EXPAND, 5)
        self.all_pro_label = wx.StaticText(output_sb_sizer.GetStaticBox(), wx.ID_ANY, u"0/0", wx.DefaultPosition,
                                           wx.DefaultSize, 0)
        self.all_pro_label.SetFont(common_font)
        self.all_pro_label.Wrap(-1)
        output_gb_sizer.Add(self.all_pro_label, wx.GBPosition(3, 17), wx.GBSpan(1, 1), wx.ALL, 5)
        output_sb_sizer.Add(output_gb_sizer, 1, wx.EXPAND, 5)

        output_gb_sizer.AddGrowableCol(4)

        self.SetSizer(output_sb_sizer)
        self.Layout()

    def __del__(self):
        pass
