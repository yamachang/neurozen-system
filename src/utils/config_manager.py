# src/utils/config_manager.py
import yaml
import os

CONFIG_FILENAME = "config.yaml"

def get_project_root() -> str:
    current_path = os.path.abspath(os.path.dirname(__file__)) # .../src/utils
    for _ in range(3): 
        # Check if config.yaml is in current_path OR if current_path contains 'src' and 'config.yaml'
        if CONFIG_FILENAME in os.listdir(current_path) or \
           (os.path.basename(current_path) == 'meditation_project' and CONFIG_FILENAME in os.listdir(current_path)) or \
           ("src" in os.listdir(current_path) and os.path.isdir(os.path.join(current_path, "src"))):
            return current_path
        current_path = os.path.dirname(current_path)
    # Fallback to CWD if project root isn't easily found by structure
    # This might happen if the script using this util is run from an unexpected location.
    # print("Warning: Project root not reliably found by structure. Falling back to CWD for config search.")
    return os.getcwd()

def load_config(config_path_override=None):
    if config_path_override and os.path.exists(config_path_override):
        config_to_load = config_path_override
    else:
        project_root = get_project_root()
        config_to_load = os.path.join(project_root, CONFIG_FILENAME)

    if not os.path.exists(config_to_load):
        # Try one level up from where this config_manager.py is (i.e. project_root if src/utils/config_manager.py)
        # This helps if get_project_root() returned a sub-optimal path.
        utils_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(utils_dir)
        project_dir_alt = os.path.dirname(src_dir)
        alt_config_path = os.path.join(project_dir_alt, CONFIG_FILENAME)
        if os.path.exists(alt_config_path):
            config_to_load = alt_config_path
        else:
            # Final fallback: current working directory (e.g., if running a script from project root)
            cwd_config_path = os.path.join(os.getcwd(), CONFIG_FILENAME)
            if os.path.exists(cwd_config_path):
                config_to_load = cwd_config_path
            else:
                raise FileNotFoundError(
                    f"Configuration file '{CONFIG_FILENAME}' not found. Tried: {config_to_load}, {alt_config_path}, and {cwd_config_path}."
                )
    
    with open(config_to_load, 'r') as f:
        config_data = yaml.safe_load(f)
    # print(f"Config loaded successfully from: {config_to_load}")
    return config_data

if __name__ == '__main__':
    try:
        config = load_config()
        print("Config loaded. Sample from EEG settings:", config.get('signal_processing', {}).get('eeg', {}).get('sample_rate'))
    except Exception as e:
        print(e)