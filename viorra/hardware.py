import os
import json
import hashlib
import time

def profile_system_hardware():
    """
    Scans the host system to determine the most powerful available GPU and its VRAM.
    This bypasses standard PyTorch/CUDA checks and queries the OS natively (WMI for Windows, 
    sysctl for Mac, lspci for Linux) to ensure Llama.cpp can latch onto the correct hardware.
    
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

def download_file_native(url, dest_path, expected_size=None, expected_sha256=None, status_callback=None):
    """
    Downloads a file using requests with stream=True to provide dead-simple progress tracking
    and strict KeyboardInterrupt (Ctrl+C) handling.
    """
    import os
    import sys
    import requests
    import hashlib
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    if expected_size and os.path.exists(dest_path) and os.path.getsize(dest_path) == expected_size:
        return True
        
    if status_callback:
        status_callback(f"Downloading {os.path.basename(dest_path)}...")
        
    try:
        headers = {}
        # Basic resume support
        if os.path.exists(dest_path):
            current_size = os.path.getsize(dest_path)
            headers['Range'] = f'bytes={current_size}-'
        else:
            current_size = 0
            
        response = requests.get(url, headers=headers, stream=True, timeout=10)
        
        # If the server doesn't support range requests, it returns 200 instead of 206
        if response.status_code == 200:
            current_size = 0
            mode = 'wb'
        elif response.status_code == 206:
            mode = 'ab'
        elif response.status_code == 416: # Range not satisfiable (already fully downloaded)
            return True
        else:
            response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0)) + current_size
        
        with open(dest_path, mode) as f:
            for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunks
                if chunk:
                    f.write(chunk)
                    current_size += len(chunk)
                    if total_size > 0:
                        pct = int((current_size / total_size) * 100)
                        # Dead-simple progress format exactly as requested by rule
                        sys.stdout.write(f"\rDownloading [{pct}%]")
                        sys.stdout.flush()
                        if status_callback:
                            status_callback(f"Downloading [{pct}%]")
                            
        sys.stdout.write("\n")
        sys.stdout.flush()
                            
    except KeyboardInterrupt:
        print("\nDownload cancelled by user (Ctrl+C). Terminating cleanly.")
        os._exit(0)
    except Exception as e:
        print(f"\nError downloading: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise

    # Verify Hash if provided to prevent silent corruption
    if expected_sha256 and os.path.exists(dest_path):
        if status_callback: status_callback("Verifying checksum...")
        hasher = hashlib.sha256()
        with open(dest_path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != expected_sha256:
            os.remove(dest_path) # Nuke corrupted file
            raise Exception("Checksum mismatch. Corrupted file deleted. Please restart.")
            
    return True

def download_llm_native(dest_path, status_callback=None):
    """Downloads the fine-tuned Viorra 2.6GB QAT GGUF model using OS-native tools."""
    url = "https://huggingface.co/qsardor/Viorra-Gemma-4-E2B-GGUF/resolve/main/gemma-4-e2b-it.Q4_K_M.gguf"
    download_file_native(url, dest_path, expected_size=None, status_callback=status_callback)
