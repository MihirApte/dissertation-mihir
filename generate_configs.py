"""
generate_configs.py
====================
Generates config YAML files for every (video x method) combination listed
in VIDEO_PROMPTS below, following the exact same settings/pattern as the
original truck/shanghai/street/dog configs (grid_size 3, sample_size 36,
pad 1, 50 inference/inversion steps).

Safe to re-run: any config file that already exists on disk is left alone
and skipped, so you can add more entries to VIDEO_PROMPTS later and re-run
this to generate just the new ones, without touching existing configs.

USAGE (run once from inside the RAVE project folder):
    python3 generate_configs.py
"""

import os

CONFIG_DIR = "configs"

# video_name -> positive prompt. Add more entries here any time you want
# to expand the test set further; re-running this script will only create
# configs for whatever's new.
VIDEO_PROMPTS = {
    "airplane":      "A vintage sepia-toned photograph of an airplane taking off",
    "baseball":      "A vibrant comic book illustration of a baseball game in action",
    "bicycle":       "A watercolor painting of a bicycle riding through a city street",
    "birds":         "A soft pastel illustration of birds flying across the sky",
    "cat":           "A children's storybook illustration of a cat walking indoors",
    "cooking":       "A warm oil painting of someone cooking in a kitchen",
    "dancer":        "A dynamic charcoal sketch of a dancer performing",
    "farmland":      "A golden-hour impressionist painting of farmland",
    "fish":          "A vivid anime-style illustration of fish swimming underwater",
    "flowers":       "A delicate watercolor painting of blooming flowers",
    "forest":        "A moody, atmospheric fantasy illustration of a forest",
    "highway":       "A neon-lit synthwave illustration of a highway at dusk",
    "interview":     "A black-and-white film noir style portrait of a person talking",
    "motorboat":     "A bright pop-art illustration of a motorboat on the ocean",
    "racecars":      "A dynamic cyberpunk illustration of race cars on a track",
    "skateboarding": "A graffiti street-art style illustration of skateboarding",
    "waterglass":    "A hyperrealistic macro painting of water in a glass",
}

# Method definitions: everything that varies between the 4 methods.
METHODS = [
    {
        "suffix": "baseline_random",
        "comment": "Baseline: random shuffle",
        "preprocess_name": "depth_zoe",
        "shuffle_mode": "random",
        "cn_scale": "1.0",
        "extra_lines": "",
    },
    {
        "suffix": "semantic_shuffle",
        "comment": "Semantic v1 shuffle (greedy NN)",
        "preprocess_name": "depth_zoe",
        "shuffle_mode": "semantic",
        "cn_scale": "1.0",
        "extra_lines": "",
    },
    {
        "suffix": "kmeans_shuffle",
        "comment": "Semantic v2 shuffle (K-means)",
        "preprocess_name": "depth_zoe",
        "shuffle_mode": "kmeans",
        "cn_scale": "1.0",
        "extra_lines": "use_freeu: false\n",
    },
    {
        "suffix": "multicontrol_random",
        "comment": "Multi-ControlNet (depth+canny) + FreeU",
        "preprocess_name": "depth_zoe-canny",
        "shuffle_mode": "random",
        "cn_scale": "'1.0-1.0'",
        "extra_lines": "use_freeu: true\n",
    },
]

TEMPLATE = """# EXPERIMENT - {video} ({comment})
video_name: "{video}"
preprocess_name: '{preprocess_name}'

batch_size: 4
batch_size_vae: 1
cond_step_start: 0.0
controlnet_conditioning_scale: {cn_scale}
controlnet_guidance_end: 1.0
controlnet_guidance_start: 0.0
give_control_inversion: true

grid_size: 3
sample_size: 36
pad: 1
guidance_scale: 7.5
inversion_prompt: ''

is_ddim_inversion: true
is_shuffle: true
shuffle_mode: '{shuffle_mode}'

negative_prompts: ""
num_inference_steps: 50
num_inversion_step: 50
positive_prompts: "{prompt}"
save_folder: '{video}_{suffix}'

{extra_lines}seed: 0
model_id: 'None'
"""


def main():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    created, skipped = 0, 0

    for video, prompt in VIDEO_PROMPTS.items():
        for method in METHODS:
            fname = f"{video}_{method['suffix']}.yaml"
            path = os.path.join(CONFIG_DIR, fname)

            if os.path.exists(path):
                print(f"[SKIP]   {fname} - already exists")
                skipped += 1
                continue

            content = TEMPLATE.format(
                video=video,
                prompt=prompt,
                comment=method["comment"],
                preprocess_name=method["preprocess_name"],
                shuffle_mode=method["shuffle_mode"],
                cn_scale=method["cn_scale"],
                suffix=method["suffix"],
                extra_lines=method["extra_lines"],
            )
            with open(path, "w") as f:
                f.write(content)
            print(f"[CREATE] {fname}")
            created += 1

    print(f"\nDone. {created} config(s) created, {skipped} already existed.")


if __name__ == "__main__":
    main()
