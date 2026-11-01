# ComfyUI 与 MiniMax H3 短剧工作流

本仓库保存本机短剧制作流程：首帧生成、H3 视频生成、中文配音与 MuseTalk 对口型，以及相关自定义节点和辅助脚本。

## 内容

- `workflows/`：可导入 ComfyUI 的工作流 JSON。
- `custom_nodes/`：流程使用的 ClipProj 和 ShortDrama 本地节点。ClipProj 的许可与说明位于其目录中。
- `pipeline/`：配音、口型处理及环境配置辅助脚本。
- `使用说明.md`、`完整短剧使用说明.md` 和 `工作流报错修复说明.md`：原电脑的操作和测试记录。

## 使用前

本仓库不包含 ComfyUI、模型权重、独立 Python 环境、示例图片或生成的视频。需要自行安装依赖和相应模型，并核对各组件与模型的许可证。现有脚本和说明包含原电脑的 `D:\ComfyUI_windows_portable`、`E:\AI\ComfyUI` 等路径，迁移时必须检查并修改；不能直接假定在其他电脑开箱即用。

在 ComfyUI 中导入 `workflows/` 下的 JSON，并按节点提示配置模型和输入素材。完整流程及本机实测范围见 `完整短剧使用说明.md`。
