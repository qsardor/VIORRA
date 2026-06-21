import os
import json
import hashlib
import time

def profile_system_hardware():
    """
    Scans the host system to determine the most powerful available GPU and its VRAM.
    This bypasses standard PyTorch/CUDA checks and queries the OS natively (WMI for Windows, 
    sysctl for Mac, lspci for Linux) to ensure LiteRT WebGPU can latch onto the correct hardware.
    
    Returns:
        tuple: (best_gpu_name (str), best_vram_in_bytes (int))
    """
    import platform
    import subprocess
    
    system = platform.system()
    best_gpu_name = None
    best_vram = 0
    
    # [WINDOWS NATIVE PROFILING]
    if system == "Windows":
        try:
            # Poll WMI for all VideoControllers to find dedicated GPUs
            out = subprocess.check_output(
                ['powershell', '-Command', 'Get-CimInstance -ClassName Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            if out:
                data = json.loads(out)
                if isinstance(data, dict):
                    data = [data]
                    
                for gpu in data:
                    name = gpu.get("Name", "")
                    ram = gpu.get("AdapterRAM") or 0
                    
                    # WMI caps AdapterRAM at 4GB (32-bit legacy integer limit).
                    # If it's an NVIDIA card, we bypass WMI and directly query nvidia-smi for true VRAM.
                    if "NVIDIA" in name.upper():
                        try:
                            smi = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW).decode().strip()
                            for mem_line in smi.splitlines():
                                true_vram = int(mem_line.strip()) * (1024**2)
                                if true_vram > ram:
                                    ram = true_vram
                        except Exception:
                            pass
                            
                    if ram > best_vram:
                        best_vram = ram
                        best_gpu_name = name
        except Exception:
            pass
            
    # [MACOS NATIVE PROFILING]
    elif system == "Darwin":
        try:
            # Apple Silicon Unified Memory is shared, so we query total system RAM
            out = subprocess.check_output(['sysctl', 'hw.memsize'], stderr=subprocess.DEVNULL).decode().strip()
            if "hw.memsize:" in out:
                ram = int(out.split(":")[1].strip())
                best_vram = ram
                best_gpu_name = "Apple Silicon GPU (Unified)"
        except Exception:
            pass
            
    # [LINUX NATIVE PROFILING]
    elif system == "Linux":
        try:
            # Use lspci to find VGA compatible controllers
            out = subprocess.check_output('lspci | grep -i "vga\\|3d\\|display"', shell=True, stderr=subprocess.DEVNULL).decode().strip()
            if out:
                for line in out.splitlines():
                    line_upper = line.upper()
                    if "NVIDIA" in line_upper:
                        best_gpu_name = "NVIDIA GPU (Linux)"
                        # Query nvidia-smi for precise VRAM allocation
                        try:
                            smi = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], stderr=subprocess.DEVNULL).decode().strip()
                            for mem_line in smi.splitlines():
                                vram = int(mem_line.strip()) * (1024**2)
                                if vram > best_vram:
                                    best_vram = vram
                        except Exception:
                            best_vram = max(best_vram, 1)
                        break
                    elif "AMD" in line_upper or "RADEON" in line_upper:
                        best_gpu_name = "AMD GPU (Linux)"
                        best_vram = max(best_vram, 1)
                        break
        except Exception:
            pass
            
    return best_gpu_name, best_vram

def download_llm_native(dest_path, status_callback=None):
    """Bypass HuggingFace Hub entirely for the massive 2.5GB model to prevent XET/LFS freezing."""
    import urllib.request
    
    url = "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm"
    expected_sha256 = "181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c"
    expected_size = 2588147712
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            existing_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
            
            if existing_size > expected_size:
                os.remove(dest_path)
                existing_size = 0
                
            headers = {'User-Agent': 'VIORRA-Native-Downloader/1.0'}
            if existing_size > 0 and existing_size < expected_size:
                headers['Range'] = f'bytes={existing_size}-'
                if status_callback: status_callback(f"Resuming download... (Attempt {attempt+1}/{max_retries})")
            elif existing_size == expected_size:
                pass # Already downloaded, skip to verification
            else:
                if status_callback: status_callback(f"Connecting... (Attempt {attempt+1}/{max_retries})")
                
            if existing_size < expected_size:
                req = urllib.request.Request(url, headers=headers)
                response = urllib.request.urlopen(req, timeout=15)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                
                # Check if server honored the Range header (206 Partial Content)
                mode = 'ab' if existing_size > 0 and response.status == 206 else 'wb'
                if mode == 'wb':
                    existing_size = 0 # Server ignored range, starting from scratch
                    
                downloaded = existing_size
                with open(dest_path, mode) as f:
                    while True:
                        chunk = response.read(1024 * 64) # 64 KB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        pct = int((downloaded / expected_size) * 100)
                        if status_callback: status_callback(f"Downloading files... [{pct}%]")
            else:
                downloaded = existing_size
                
            # If download loop finishes or was already finished, verify full file hash
            if downloaded == expected_size:
                if status_callback: status_callback("Verifying checksum...")
                hasher = hashlib.sha256()
                with open(dest_path, 'rb') as f:
                    # Hash in 1MB chunks so we don't blow up system RAM reading 2.5GB at once
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        hasher.update(chunk)
                
                actual_sha256 = hasher.hexdigest()
                if actual_sha256 == expected_sha256:
                    return # Download successful and verified!
                else:
                    # Hash failed. It's corrupted. Delete and restart the entire download on next loop
                    os.remove(dest_path)
                    if attempt == max_retries - 1:
                        raise Exception(f"Checksum mismatch! Expected {expected_sha256}, got {actual_sha256}")
            else:
                # Connection dropped mid-way. Loop will retry and RESUME using the new file size.
                time.sleep(2)
                
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Native LLM download failed after {max_retries} attempts: {e}")
            time.sleep(2)
