
import sys
import os
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

PASS = "  [PASS]"
FAIL = "  [FAIL]"

# All keys the pipeline reads in __call__
REQUIRED_KEYS = [
    'video_name',
    'preprocess_name',
    'batch_size',
    'batch_size_vae',
    'cond_step_start',
    'controlnet_conditioning_scale',
    'controlnet_guidance_end',
    'controlnet_guidance_start',
    'give_control_inversion',
    'grid_size',
    'sample_size',
    'pad',
    'guidance_scale',
    'inversion_prompt',
    'is_ddim_inversion',
    'is_shuffle',
    'negative_prompts',
    'num_inference_steps',
    'num_inversion_step',
    'positive_prompts',
    'save_folder',
    'seed',
    'model_id',
    'shuffle_mode',     # our new parameter
]

VALID_PREPROCESSORS = [
    'lineart_realistic', 'lineart_coarse', 'lineart_standard',
    'lineart_anime', 'lineart_anime_denoise',
    'softedge_hed', 'softedge_hedsafe', 'softedge_pidinet', 'softedge_pidsafe',
    'canny',
    'depth_leres', 'depth_leres++', 'depth_midas', 'depth_zoe',
]

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs')


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def test_config_has_all_required_keys(cfg, name):
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    assert not missing, f"{name}: missing keys: {missing}"
    print(PASS, f"{name}: all {len(REQUIRED_KEYS)} required keys present")


def test_config_shuffle_mode_valid(cfg, name):
    mode = cfg.get('shuffle_mode')
    assert mode in ('random', 'semantic'), \
        f"{name}: shuffle_mode must be 'random' or 'semantic', got '{mode}'"
    print(PASS, f"{name}: shuffle_mode = '{mode}' (valid)")


def test_config_preprocessor_valid(cfg, name):
    pre = cfg.get('preprocess_name', '')
    # Multi-controlnet: names joined by '-'
    parts = pre.split('-') if '-' in pre else [pre]
    for part in parts:
        assert part in VALID_PREPROCESSORS, \
            f"{name}: unknown preprocessor '{part}'"
    print(PASS, f"{name}: preprocess_name = '{pre}' (valid)")


def test_config_numeric_ranges(cfg, name):
    assert cfg['guidance_scale'] > 0,         f"{name}: guidance_scale must be > 0"
    assert cfg['num_inference_steps'] > 0,    f"{name}: num_inference_steps must be > 0"
    assert cfg['num_inversion_step'] > 0,     f"{name}: num_inversion_step must be > 0"
    assert cfg['grid_size'] >= 2,             f"{name}: grid_size must be >= 2"
    assert cfg['batch_size'] >= 1,            f"{name}: batch_size must be >= 1"
    assert cfg['pad'] >= 1,                   f"{name}: pad must be >= 1"
    assert 0.0 <= cfg['controlnet_guidance_start'] <= 1.0
    assert 0.0 <= cfg['controlnet_guidance_end']   <= 1.0
    print(PASS, f"{name}: all numeric values in valid ranges")


def test_baseline_is_random(cfg):
    assert cfg['shuffle_mode'] == 'random', \
        f"baseline_random.yaml should have shuffle_mode='random', got '{cfg['shuffle_mode']}'"
    print(PASS, "baseline_random.yaml: shuffle_mode is 'random'")


def test_semantic_is_semantic(cfg):
    assert cfg['shuffle_mode'] == 'semantic', \
        f"semantic_shuffle.yaml should have shuffle_mode='semantic', got '{cfg['shuffle_mode']}'"
    print(PASS, "semantic_shuffle.yaml: shuffle_mode is 'semantic'")


def test_both_configs_use_same_video_and_preprocessor(baseline, semantic):
    assert baseline['video_name'] == semantic['video_name'], \
        "Both configs must use the same video for a fair comparison"
    assert baseline['preprocess_name'] == semantic['preprocess_name'], \
        "Both configs must use the same preprocessor for a fair comparison"
    assert baseline['grid_size'] == semantic['grid_size'], \
        "Both configs must use the same grid_size for a fair comparison"
    assert baseline['seed'] == semantic['seed'], \
        "Both configs must use the same seed for a fair comparison"
    print(PASS, "Both configs: same video, preprocessor, grid_size, seed (fair comparison)")


def test_save_folders_differ(baseline, semantic):
    assert baseline['save_folder'] != semantic['save_folder'], \
        "Configs must have different save_folders or outputs will overwrite each other"
    print(PASS, "Both configs: different save_folders (outputs won't overwrite)")


if __name__ == '__main__':
    print("\n-- Config Validation Tests -------------------------------")

    baseline_path = os.path.join(CONFIGS_DIR, 'baseline_random.yaml')
    semantic_path = os.path.join(CONFIGS_DIR, 'semantic_shuffle.yaml')

    passed = 0
    failed = 0

    def run(fn, *args):
        global passed, failed
        try:
            fn(*args)
            passed += 1
        except Exception as e:
            print(f"  [FAIL]  {fn.__name__}: {e}")
            failed += 1

    # Load configs
    try:
        baseline = load_yaml(baseline_path)
        print(f"  Loaded: {baseline_path}")
    except Exception as e:
        print(f"  [FAIL]  Could not load baseline_random.yaml: {e}")
        sys.exit(1)

    try:
        semantic = load_yaml(semantic_path)
        print(f"  Loaded: {semantic_path}")
    except Exception as e:
        print(f"  [FAIL]  Could not load semantic_shuffle.yaml: {e}")
        sys.exit(1)

    print()

    # Per-config tests
    for cfg, name in [(baseline, 'baseline_random.yaml'),
                      (semantic, 'semantic_shuffle.yaml')]:
        run(test_config_has_all_required_keys, cfg, name)
        run(test_config_shuffle_mode_valid, cfg, name)
        run(test_config_preprocessor_valid, cfg, name)
        run(test_config_numeric_ranges, cfg, name)

    # Cross-config tests
    run(test_baseline_is_random,  baseline)
    run(test_semantic_is_semantic, semantic)
    run(test_both_configs_use_same_video_and_preprocessor, baseline, semantic)
    run(test_save_folders_differ, baseline, semantic)

    print(f"\n  {passed}/{passed+failed} tests passed")
    if failed > 0:
        sys.exit(1)
