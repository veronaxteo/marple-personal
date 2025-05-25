import json
import csv
import shutil


info_file = "../../../data/exp2/detective_trial_info.csv"

with open(info_file, "r") as f:
    reader = csv.DictReader(f)
    trial_num = 1
    for row in reader:
        in_file = f"../trials_v2_suspect/json/{row['trial']}_A1.json"
        with open(in_file, "r") as inf:
            trial = json.load(inf)
        # before image
        shutil.copyfile(in_file, f"json/{trial_num}_1.json")
        shutil.copyfile(f"../trials_v2_suspect/images/{row['trial']}_A1.png",
                        f"images/{trial_num}_1.png")
        # after image
        kitchen = list(filter(lambda r: r["type"] == "Kitchen", trial["Grid"]["rooms"]["initial"]))[0]
        kitchen["furnitures"]["num"] += 1
        crumb = {
            "type": "crumbs",
            "pos": list(eval(row["crumb_location"])),
            "objs": {"num": 0, "initial": []}
        }
        kitchen["furnitures"]["initial"].append(crumb)
        out_file = f"json/{trial_num}_2.json"
        with open(out_file, "w") as outf:
            outf.write(json.dumps(trial, indent=4))
        trial_num += 1
