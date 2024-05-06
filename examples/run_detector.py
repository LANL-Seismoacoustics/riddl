# © 2024. Triad National Security, LLC. All rights reserved.
"""
Load infrapy run_fk results and riddl model to detect siganls.
"""
from riddl.models import fk

if __name__ == '__main__':
    # Data directory and output ID
    model_id = "./models/use/soi2"
    fk_label = "./data/Bishop_etal2022/Fig5/fig5.fk_results.dat"

    # Run analysis and summarize results
    det_times, predictions = fk.run(fk_label, model_id)
    for n in range(0, len(det_times)):
        print(det_times[n], '\t', predictions[n])

    fk.plot(fk_label, det_times, predictions)