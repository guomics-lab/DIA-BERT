import os

import wx
import yaml


# 读取yml配置文件
def read_yml():
    cwd = os.getcwd()
    yaml_file = os.path.join(cwd, 'config/gui_config.yml')
    content = None
    with open(yaml_file, 'r', encoding='utf-8') as stream:
        try:
            content = yaml.safe_load(stream)
        except yaml.YAMLError as err:
            raise yaml.YAMLError("The yaml file {} could not be parsed. {}".format(yaml_file, err))
    return content

gui_yml = read_yml()

introduce_text = gui_yml['about']['introduce']['text']

common_font = wx.Font()
introduce_font = wx.Font()
version_font = wx.Font()

common_font.SetFamily(gui_yml['font']['family'])
common_font.SetWeight(gui_yml['font']['weight'])
common_font.SetFaceName(gui_yml['font']['faceName'])
common_font.SetPointSize(gui_yml['font']['size'])


introduce_font.SetFamily(gui_yml['font']['family'])
introduce_font.SetWeight(gui_yml['about']['introduce']['weight'])
introduce_font.SetFaceName(gui_yml['font']['faceName'])
introduce_font.SetPointSize(gui_yml['about']['introduce']['fontSize'])

version_font.SetFamily(gui_yml['font']['family'])
version_font.SetWeight(gui_yml['about']['version']['weight'])
version_font.SetFaceName(gui_yml['font']['faceName'])
version_font.SetPointSize(gui_yml['about']['version']['fontSize'])

