import os.path
from src.utils.gpu_utils import get_top_free_device

import torch
import wx

from gui.panel.run_panel import RunInfoPanel
from src import common_config
from src.common import constant
from src.common.constant import ProgressStepStatusEnum
from src.common.obj import InputParam
from src.common.runtime_data_info import runtime_data
from src.threads.identify_thread import IdentifyThread
from src.utils import msg_send_utils, pkl_size_util


class RunPanelEventHandler(object):

    def __init__(self, run_info_panel: RunInfoPanel):
        self.run_info_panel = run_info_panel

    def lib_btn_click(self, event):
        filesFilter = "*.tsv; *.csv"
        file_choose_bt = wx.FileDialog(self.run_info_panel, message='Choose library', wildcard=filesFilter,
                                       style=wx.FD_DEFAULT_STYLE)
        if file_choose_bt.ShowModal() == wx.ID_OK:
            self.run_info_panel.config_panel.lib_path_text.ChangeValue(file_choose_bt.GetPath())
        file_choose_bt.Destroy()

    def mzml_choose_click(self, event):
        filesFilter = "*.mzML"
        file_choose_bt = wx.FileDialog(self.run_info_panel, message='Choose mzML', wildcard=filesFilter, style=wx.FD_MULTIPLE)
        if file_choose_bt.ShowModal() == wx.ID_OK:
            self.run_info_panel.input_panel.mzml_data_list.DeleteAllItems()
            runtime_data.mzml_list = []
            choose_file_path_list = file_choose_bt.GetPaths()
            for dd, choose_file_path in enumerate(choose_file_path_list):
                indexItem = self.run_info_panel.input_panel.mzml_data_list.InsertItem(len(choose_file_path_list), choose_file_path)
                runtime_data.mzml_list.append(choose_file_path)
        file_choose_bt.Destroy()

    def clear_btn_click(self, event):
        self.run_info_panel.input_panel.mzml_data_list.DeleteAllItems()
        runtime_data.mzml_list = []

    def output_dir_choose(self, event):
        dir_choose_bt = wx.DirDialog(self.run_info_panel, 'Choose output dir path', style=wx.DD_DIR_MUST_EXIST)
        if dir_choose_bt.ShowModal() == wx.ID_OK:
            self.run_info_panel.input_panel.file_output_dir_text.ChangeValue(dir_choose_bt.GetPath())
        dir_choose_bt.Destroy()

    def run_btn_click(self, event):
        # 检测有没有cuda
        if not torch.cuda.is_available():
            msg_box = wx.MessageDialog(None, 'Sorry, there is no GPU on this machine.', 'alert', wx.YES_DEFAULT | wx.ICON_ERROR)
            if msg_box.ShowModal() == wx.ID_YES:
                msg_box.Destroy()
            return
        lib_path = self.run_info_panel.config_panel.lib_path_text.GetValue()
        if not lib_path or not os.path.exists(lib_path):
            msg_box = wx.MessageDialog(None, 'Spectral lib is not exist.', 'alert', wx.YES_DEFAULT | wx.ICON_ERROR)
            if msg_box.ShowModal() == wx.ID_YES:
                msg_box.Destroy()
            return

        if not runtime_data.mzml_list:
            msg_box = wx.MessageDialog(None, 'Please choose identify mzML.', 'alert', wx.YES_DEFAULT | wx.ICON_ERROR)
            if msg_box.ShowModal() == wx.ID_YES:
                msg_box.Destroy()
            return

        output_path = self.run_info_panel.input_panel.file_output_dir_text.GetValue()
        if not output_path:
            msg_box = wx.MessageDialog(None, 'Output dir is not set.', 'alert', wx.YES_DEFAULT | wx.ICON_ERROR)
            if msg_box.ShowModal() == wx.ID_YES:
                msg_box.Destroy()
            return

        input_param = self.build_input_param()
        # 更新process info相关
        self.init_progress_info()
        self.update_btn_running()
        self.disable_btn()
        runtime_data.identify_thread = IdentifyThread(input_param, constant.msg_queue)
        runtime_data.identify_thread.start()


    def stop_btn_click(self, event):
        if runtime_data.identify_thread:
            msg_send_utils.send_msg(status=ProgressStepStatusEnum.STOPPING, channel=constant.main_msg_channel)
            runtime_data.running_flag = False
            runtime_data.identify_thread.terminate()
            runtime_data.identify_thread.join()
            msg_send_utils.send_msg(status=ProgressStepStatusEnum.STOPPED, msg='Processing stopped', channel=constant.main_msg_channel)
            runtime_data.identify_thread = None

    '''
    构建input param
    '''
    def build_input_param(self) -> InputParam:
        input_param = InputParam()
        lib_path = self.run_info_panel.config_panel.lib_path_text.GetValue()
        output_path = self.run_info_panel.input_panel.file_output_dir_text.GetValue()
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        raw_txt_path = self.build_run_mzml_txt(output_path)
        input_param.rawdata_file_dir_path = raw_txt_path
        input_param.lib = lib_path
        input_param.out_path = output_path
        input_param.env = constant.env_win
        input_param.device = 'cuda'
        input_param.score_device = 'cuda'

        n_thread = self.run_info_panel.config_panel.thread_num_spin_ctrl.GetValue()
        input_param.n_thread = n_thread

        max_gpu_num = self.run_info_panel.config_panel.gpu_num_spin_ctrl.GetValue()
        # 先计算每个GPU的使用率，然后从大到小排序
        topn_device_list, topn_min_free = get_top_free_device(max_gpu_num)
        input_param.gpu_devices = ','.join(topn_device_list)


        input_param.train_pkl_size = pkl_size_util.calculate_pkl_size(topn_min_free)
        input_param.quant_pkl_size = pkl_size_util.calculate_pkl_size(topn_min_free)

        decoy_method_select_id = self.run_info_panel.config_panel.decoy_method_choice.GetSelection()
        input_param.decoy_method = constant.decoy_method_list[decoy_method_select_id]
        input_param.fitting_rt_num = self.run_info_panel.config_panel.rt_norm_spin_ctrl.GetValue()
        input_param.fitting_rt_batch_size = self.run_info_panel.config_panel.fitting_rt_batch_size_spin_ctrl.GetValue()

        mz_rt_unit_select_id = self.run_info_panel.config_panel.mz_rt_unit_choice.GetSelection()
        input_param.mz_unit = constant.mz_rt_unit_list[mz_rt_unit_select_id]

        raw_rt_unit_select_id = self.run_info_panel.config_panel.rt_unit_choice.GetSelection()
        input_param.raw_rt_unit = constant.raw_rt_unit_list[raw_rt_unit_select_id]

        instrument_select_id = self.run_info_panel.config_panel.instrument_choice.GetSelection()
        input_param.instrument = constant.instrument_list[instrument_select_id]

        batch_size = self.run_info_panel.config_panel.batch_size_spin_ctrl.GetValue()
        input_param.batch_size = batch_size

        input_param.step_size = self.run_info_panel.config_panel.batch_score_size_spin_ctrl.GetValue()


        protein_infer_key_select_id = self.run_info_panel.config_panel.protein_infer_choice.GetSelection()
        input_param.protein_infer_key = constant.protein_infer_key_list[protein_infer_key_select_id]

        # n_cycles = self.run_info_panel.config_panel.rt_width_spin_ctrl.GetValue()
        finetune_score_limit = self.run_info_panel.config_panel.finetune_score_spin_ctrl.GetValue()
        input_param.finetune_score_limit = finetune_score_limit

        train_epochs = self.run_info_panel.config_panel.train_epochs_spin_ctrl.GetValue()
        input_param.train_epochs = train_epochs

        common_config_data = common_config.read_yml()
        input_param.n_cycles = common_config_data['identify']['n_cycles']
        input_param.max_fragment = common_config_data['identify']['max_fragment']
        input_param.iso_range = common_config_data['identify']['iso_range']
        input_param.mz_min = common_config_data['identify']['mz_min']
        input_param.mz_max = common_config_data['identify']['mz_max']
        input_param.seed = common_config_data['identify']['seed']
        input_param.model_cycles = common_config_data['identify']['model_cycles']
        input_param.frag_repeat_num = common_config_data['identify']['frag_repeat_num']
        input_param.xrm_model_file = common_config_data['identify']['xrm_model_file']

        input_param.dev_model = 1
        input_param.skip_no_temp = 0

        # input_param.open_base_identify = False
        # input_param.open_finetune_peak = False
        # input_param.clear_data = False

        # 获取checkbox的选择状态
        open_lib_decoy = self.run_info_panel.config_panel.lib_decoy_check_box.GetValue()
        input_param.open_lib_decoy = bool(open_lib_decoy)

        open_identify = self.run_info_panel.config_panel.identify_check_box.GetValue()
        input_param.open_identify = bool(open_identify)

        open_quantification = self.run_info_panel.config_panel.cross_run_check_box.GetValue()
        input_param.open_quantification = bool(open_quantification)

        return input_param

    def init_progress_info(self):
        self.run_info_panel.run_info_display_panel.progress_data_list.DeleteAllItems()
        for dd, choose_file_path in enumerate(runtime_data.mzml_list):
            indexItem = self.run_info_panel.run_info_display_panel.progress_data_list.InsertItem(dd, os.path.split(choose_file_path)[-1])
            self.run_info_panel.run_info_display_panel.progress_data_list.SetItem(indexItem, 1, 'wait')
            self.run_info_panel.run_info_display_panel.progress_data_list.SetItem(indexItem, 2, 'wait')
            self.run_info_panel.run_info_display_panel.progress_data_list.SetItem(indexItem, 3, 'wait')
            self.run_info_panel.run_info_display_panel.progress_data_list.SetItem(indexItem, 4, 'wait')
            self.run_info_panel.run_info_display_panel.progress_data_list.SetItem(indexItem, 5, 'wait')
            self.run_info_panel.run_info_display_panel.progress_data_list.SetItem(indexItem, 6, 'wait')
            self.run_info_panel.run_info_display_panel.progress_data_list.SetItem(indexItem, 7, 'wait')
        self.run_info_panel.run_info_display_panel.all_progress_gauge.SetValue(0)
        self.run_info_panel.run_info_display_panel.all_progress_gauge.SetRange(len(runtime_data.mzml_list))

        self.run_info_panel.run_info_display_panel.all_pro_label.SetLabel('{}/{}'.format(0, len(runtime_data.mzml_list)))

    '''
    更新相关按钮状态信息
    '''
    def update_btn_running(self):
        self.run_info_panel.log_panel.log_text.SetValue('')
        self.run_info_panel.run_control_panel.run_button.Disable()
        self.run_info_panel.run_control_panel.run_status_button.SetBackgroundColour(constant.RUNNING_COLOR)
        self.run_info_panel.run_control_panel.run_status_label.SetLabel('Running')
        self.run_info_panel.run_control_panel.stop_button.Enable()

    def disable_btn(self):
        self.run_info_panel.config_panel.lib_btn.Disable()
        self.run_info_panel.input_panel.mzml_select_button.Disable()

    def build_run_mzml_txt(self, output_path):
        raw_txt_path = os.path.join(output_path, 'run_mzml.txt')
        with open(raw_txt_path, mode='w+') as f:
            for mzml_path in runtime_data.mzml_list:
                f.write('{}\n'.format(mzml_path))
        return raw_txt_path

