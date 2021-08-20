yourpath = 'dataset/sjajni'

import os
for root, dirs, files in os.walk(yourpath, topdown=True):
    for name in files:
        print("Name = ", root)
        if name == "results.txt":
            f_results = open(os.path.join(root, name))
            new_dir = os.path.join("./dataset/"+root[11:])
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            f_data = open(os.path.join(new_dir, "data.txt"), "w")

            for i, line in enumerate(f_results):
                if i == 15:
                    f_data.writelines(line)
                elif i == 16:
                    f_data.writelines(line)
                elif i == 18:
                    f_data.writelines(line)
                elif i == 19:
                    f_data.writelines(line)

            f_data.close()
            f_results.close()


