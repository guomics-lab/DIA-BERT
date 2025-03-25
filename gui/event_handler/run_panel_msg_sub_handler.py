import json

from gui.panel.run_panel import RunInfoPanel
from src.common import constant
from src.common import runtime_data_info
from src.common.obj import IdentifyMsg


class RunPanelMsgSubHandler(object):

    def __init__(self, run_info_panel: RunInfoPanel):
        self.run_info_panel = run_info_panel

    def sub_msg(self, msg):
        # IdentifyMsg
        pdl = self.run_info_panel.run_info_display_panel.progress_data_list
        identify_msg_info = json.loads(msg, object_hook=IdentifyMsg.json_to_object)
        raw_index = identify_msg_info.mzml_index
        if identify_msg_info.status == constant.ProgressStepStatusEnum.RUNNING:
            # 把节点的状态修改为running
            # 记录一下当前处理的文件的位置
            if identify_msg_info.step == constant.ProgressStepEnum.PARSE_MZML:
                pdl.SetItem(raw_index, 1, 'Running')
            elif identify_msg_info.step == constant.ProgressStepEnum.RT_NORMALIZATION:
                pdl.SetItem(raw_index, 2, 'Running')
            elif identify_msg_info.step == constant.ProgressStepEnum.SCREEN:
                pdl.SetItem(raw_index, 3, 'Running')
            elif identify_msg_info.step == constant.ProgressStepEnum.PREPARE_DATA:
                pdl.SetItem(raw_index, 4, 'Running')
            elif identify_msg_info.step == constant.ProgressStepEnum.FINETUNE_TRAIN:
                pdl.SetItem(raw_index, 5, 'Running')
            elif identify_msg_info.step == constant.ProgressStepEnum.FINETUNE_EVAL:
                pdl.SetItem(raw_index, 6, 'Running')
            elif identify_msg_info.step == constant.ProgressStepEnum.QUANT:
                pdl.SetItem(raw_index, 7, 'Running')
        elif identify_msg_info.status == constant.ProgressStepStatusEnum.SUCCESS:
            # 把节点的状态修改为success
            if identify_msg_info.step == constant.ProgressStepEnum.PARSE_MZML:
                pdl.SetItem(raw_index, 1, 'Success')
            elif identify_msg_info.step == constant.ProgressStepEnum.RT_NORMALIZATION:
                pdl.SetItem(raw_index, 2, 'Success')
            elif identify_msg_info.step == constant.ProgressStepEnum.SCREEN:
                pdl.SetItem(raw_index, 3, 'Success')
            elif identify_msg_info.step == constant.ProgressStepEnum.PREPARE_DATA:
                pdl.SetItem(raw_index, 4, 'Success')
            elif identify_msg_info.step == constant.ProgressStepEnum.FINETUNE_TRAIN:
                pdl.SetItem(raw_index, 5, 'Success')
            elif identify_msg_info.step == constant.ProgressStepEnum.FINETUNE_EVAL:
                pdl.SetItem(raw_index, 6, 'Success')
            elif identify_msg_info.step == constant.ProgressStepEnum.QUANT:
                pdl.SetItem(raw_index, 7, 'Success')
        elif identify_msg_info.status == constant.ProgressStepStatusEnum.ERROR:
            # 把节点的状态修改为 error
            # 把节点的状态修改为success
            if identify_msg_info.step == constant.ProgressStepEnum.PARSE_MZML:
                pdl.SetItem(raw_index, 1, 'Error')
            elif identify_msg_info.step == constant.ProgressStepEnum.RT_NORMALIZATION:
                pdl.SetItem(raw_index, 2, 'Error')
            elif identify_msg_info.step == constant.ProgressStepEnum.SCREEN:
                pdl.SetItem(raw_index, 3, 'Error')
            elif identify_msg_info.step == constant.ProgressStepEnum.PREPARE_DATA:
                pdl.SetItem(raw_index, 4, 'Error')
            elif identify_msg_info.step == constant.ProgressStepEnum.FINETUNE_TRAIN:
                pdl.SetItem(raw_index, 5, 'Error')
            elif identify_msg_info.step == constant.ProgressStepEnum.FINETUNE_EVAL:
                pdl.SetItem(raw_index, 6, 'Error')
            elif identify_msg_info.step == constant.ProgressStepEnum.QUANT:
                pdl.SetItem(raw_index, 7, 'Error')
        elif identify_msg_info.status == constant.ProgressStepStatusEnum.IDENTIFY_NUM:
            pdl.SetItem(raw_index, 3, 'Running({}/{})'.format(runtime_data_info.runtime_data.current_identify_num, runtime_data_info.runtime_data.current_identify_all_num))
        elif identify_msg_info.status == constant.ProgressStepStatusEnum.END:
            # 进度条 + 1
            self.run_info_panel.run_info_display_panel.all_progress_gauge.SetValue(runtime_data_info.runtime_data.current_mzml_index + 1)
            self.run_info_panel.run_info_display_panel.all_pro_label.SetLabel('{}/{}'.format(runtime_data_info.runtime_data.current_mzml_index + 1, len(runtime_data_info.runtime_data.mzml_list)))
            # 判断如果是失败的，就标记为红色
            pdl.SetItemTextColour(raw_index, constant.OVER_COLOR)

        elif identify_msg_info.status == constant.ProgressStepStatusEnum.FAIL_END:
            # 进度条 + 1
            self.run_info_panel.run_info_display_panel.all_progress_gauge.SetValue(runtime_data_info.runtime_data.current_mzml_index + 1)
            self.run_info_panel.run_info_display_panel.all_pro_label.SetLabel('{}/{}'.format(runtime_data_info.runtime_data.current_mzml_index + 1, len(runtime_data_info.runtime_data.mzml_list)))
            # 判断如果是失败的，就标记为红色

            pdl.SetItemTextColour(raw_index, constant.ERROR_COLOR)

        elif identify_msg_info.status == constant.ProgressStepStatusEnum.STOPPING:
            self.update_btn_stopping()
        elif identify_msg_info.status == constant.ProgressStepStatusEnum.STOPPED:
            self.update_btn_stopped()
            self.enable_btn()
        elif identify_msg_info.status == constant.ProgressStepStatusEnum.ALL_END:
            self.update_btn_finished()
            self.enable_btn()
        if identify_msg_info.msg:
            self.run_info_panel.log_panel.log_text.AppendText(identify_msg_info.msg + '\n')

    def update_btn_stopping(self):
        self.run_info_panel.run_control_panel.run_button.Disable()
        self.run_info_panel.run_control_panel.run_status_button.SetBackgroundColour(constant.RUNNING_COLOR)
        self.run_info_panel.run_control_panel.run_status_label.SetLabel('Stopping')
        self.run_info_panel.run_control_panel.stop_button.Enable()


    def update_btn_finished(self):
        self.run_info_panel.run_control_panel.run_button.Enable()
        self.run_info_panel.run_control_panel.run_status_button.SetBackgroundColour(constant.OVER_COLOR)
        self.run_info_panel.run_control_panel.run_status_label.SetLabel('Finished')
        self.run_info_panel.run_control_panel.stop_button.Disable()

    def update_btn_stopped(self):
        self.run_info_panel.run_control_panel.run_button.Enable()
        self.run_info_panel.run_control_panel.run_status_button.SetBackgroundColour(None)
        self.run_info_panel.run_control_panel.run_status_label.SetLabel('Stopped')
        self.run_info_panel.run_control_panel.stop_button.Disable()

    def enable_btn(self):
        self.run_info_panel.config_panel.lib_btn.Enable()
        self.run_info_panel.input_panel.mzml_select_button.Enable()
