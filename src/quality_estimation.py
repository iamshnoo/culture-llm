#

import json
from comet import download_model, load_from_checkpoint

model_path = download_model(
    "Unbabel/wmt22-cometkiwi-da",
    saving_directory="/projects/cometkiwi/da",
)

with open("../data/qe.json") as f:
    all_data = json.load(f)

lang = "sw"  # can be "zh", "el", "hi", "fa", "sw"
field = "instructions"  # can be "instructions", "inputs", "outputs"
data = all_data[lang][field]

for item in data:
    item["src"] = str(item["src"])
    item["mt"] = str(item["mt"])

model = load_from_checkpoint(model_path)
model_output = model.predict(data, batch_size=8, gpus=1)
print(model_output["system_score"])
