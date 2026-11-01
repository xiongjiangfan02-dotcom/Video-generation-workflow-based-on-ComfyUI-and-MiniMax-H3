$ErrorActionPreference = 'Stop'
$root = 'E:\AI\ComfyUI'
$python = "$root\pipeline\python310\python.exe"
$env:PIP_CACHE_DIR = "$root\cache\pip"
$env:HF_HOME = "$root\cache\huggingface"
$env:TORCH_HOME = "$root\cache\torch"
$env:TEMP = "$root\cache\temp"
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'
$env:PATH = "$root\pipeline\python310\Scripts;$root\pipeline\ffmpeg;$env:PATH"
$wheelNames = @(
    'torch-2.0.1+cu118-cp310-cp310-win_amd64.whl',
    'torchvision-0.15.2+cu118-cp310-cp310-win_amd64.whl',
    'torchaudio-2.0.2+cu118-cp310-cp310-win_amd64.whl'
)
$wheels = @($wheelNames | ForEach-Object { "$root\pipeline\wheels\$_" })
while (@($wheels | Where-Object { -not (Test-Path -LiteralPath $_) }).Count -gt 0) {
    Write-Output 'WAITING_FOR_VERIFIED_CUDA_WHEELS'
    Start-Sleep -Seconds 30
}
& $python -m pip install @wheels --index-url https://pypi.tuna.tsinghua.edu.cn/simple --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'CUDA runtime package installation failed.' }
& $python -m pip install 'https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/mmcv-2.0.1-cp310-cp310-win_amd64.whl' -c "$root\pipeline\inference-requirements.txt" --index-url https://pypi.tuna.tsinghua.edu.cn/simple --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'MMCV installation failed.' }
& $python -m pip install --no-build-isolation -r "$root\pipeline\inference-requirements.txt" --index-url https://pypi.tuna.tsinghua.edu.cn/simple --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'MuseTalk dependencies installation failed.' }
& $python -c "import torch,mmcv,mmpose,sherpa_onnx; from mmcv.ops import nms; print('CUDA:',torch.cuda.is_available()); print('TORCH:',torch.__version__); print('MMCV:',mmcv.__version__); print('MMPOSE:',mmpose.__version__); assert torch.cuda.is_available()"
if ($LASTEXITCODE -ne 0) { throw 'Runtime validation failed.' }
Write-Output 'ISOLATED_RUNTIME_VERIFIED'
