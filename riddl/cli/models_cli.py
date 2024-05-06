# © 2024. Triad National Security, LLC. All rights reserved.
import click
import json
import numpy as np
from riddl.models import fk
from riddl.utils import data_io
from tensorflow.keras.models import load_model

# Type conversion function for json serialization
def default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Not serializable')


@click.command('build', short_help="Build tensors from a directory of `.npy` files.")
@click.option("--data-dir", help="Data directory (required)", prompt="prompt data_dir")
@click.option("--output-id", help="Output ID label", default=None)
@click.option("--output-dir", help="Output directory", default=None)
@click.option("--labels-file", help="Text file of labels", default=None)
@click.option("--test-fraction", help="Fraction of data to use for testing", default=0.25)
@click.option("--merge-labels-file", help="Text file of labels to merge", default=None)
@click.option("--file_pattern", help="Text file of patterns to filter inputs", default=None)
def build(data_dir: str, output_id: str, output_dir: str, labels_file: str, test_fraction: float, merge_labels_file: str, file_pattern: str) -> None:
    '''
    \b
    riddl models fk build
    --------------------
    \b
    Example Usage:
    (from the `examples' folder)
    \t riddl models fk build --data-dir ./data/Bishop_etal2022 --output-id soi_example --output-dir ./models/example --labels-file labels_file.txt --merge-labels-file merge_labels_file.txt
    '''

    click.echo("")
    click.echo("#############################################")
    click.echo("##                                         ##")
    click.echo("##                    riddl                ##")
    click.echo("##            AI/ML Model Methods          ##")
    click.echo("##                                         ##")
    click.echo("##      Build tensors from .npy files      ##")
    click.echo("##                                         ##")
    click.echo("#############################################")
    click.echo("")

    # Check for labels file and remove an EOL characters
    if labels_file:
        with open(labels_file) as lab_file:
            label_list = lab_file.readlines()
        label_list = [item.rstrip("\n") for item in label_list]
    else:
        label_list = None

    # Check for labels file and remove an EOL characters
    if merge_labels_file:
        with open(merge_labels_file) as merge_file:
            merge_list = list(merge_file.readlines())
        merge_list = [item.rstrip("\n") for item in merge_list]
    else:
        merge_list = None

    if file_pattern:
        with open(file_pattern) as pattern_file:
            pattern_list = list(pattern_file.readlines())
        pattern_list = [item.rstrip("\n") for item in pattern_list]
    else:
        pattern_list = None

    # Read in file list
    file_list, _ = data_io.build_file_list(data_dir, labels=label_list, folders=label_list, file_pattern=pattern_list)

    # Build tensors from data lists - data preparation functions
    output_id = output_dir + "/" + output_id
    data_io.write_tensor_data(file_list, output_id, labels=label_list, test_fraction=test_fraction, merge_labels=merge_list)


@click.command('train', short_help="Train and evaluate a detection model")
@click.option("--data-id", help="Directory of atmospheric specifications (required)", prompt="prompt data_id")
@click.option("--num_folds", help="Directory of atmospheric specifications (required)", default=1)
@click.option("--epoch_num", help="Directory of atmospheric specifications (required)", default=2)
@click.option("--batch_size", help="Directory of atmospheric specifications (required)", default=32)
@click.option("--model_out", help="Name of trained model", default="model_out_default_name")
def train(data_id: str, num_folds: int, epoch_num: int, batch_size: int, model_out: str) -> None:
    '''
    \b
    riddl models fk train
    --------------------
    \b
    Example Usage:
    (from the `examples' folder)
    \t riddl models fk train  --data-id ./models/example/soi_example --num_folds 1
    '''

    click.echo("")
    click.echo("#############################################")
    click.echo("##                                         ##")
    click.echo("##                    riddl                ##")
    click.echo("##            AI/ML Model Methods          ##")
    click.echo("##                                         ##")
    click.echo("##   Train and save a fk detection model   ##")
    click.echo("##                                         ##")
    click.echo("#############################################")
    click.echo("")

    if num_folds == 1:
        data_id_train = data_id + "_train"
        data_id_test = data_id + "_test"

        X_train, Y_train, _ = data_io.load_tensor_data(data_id_train)
        X_test, Y_test, _ = data_io.load_tensor_data(data_id_test)

        print("Training a model without a k-fold split.")

        # Train model
        model = fk.train_ML_model(X_train, Y_train, epoch_cnt=epoch_num, batch_size=batch_size)
        # Evaluate trained model
        score, M = fk.evaluate_model(X_test, Y_test, model=model, batch_size=batch_size)
        # Save model
        model.save(model_out)

        precision = np.array([M[n, n] / np.sum(M[:, n]) for n in range(M.shape[0])])
        recall = np.array([M[n, n] / np.sum(M[n, :]) for n in range(M.shape[1])])
        fscore = 2 * (precision * recall) / (precision + recall)

        print("Training summary:")
        print("Accuracy: " + str(score[1] * 100))
        print("Loss: " + str(score[0]))
        print('\n' + "Confusion Matrix (true labels along horizontals) [%]: ", '\n', M.T)
        print("\nPrecision = " + str(precision))
        print("\nRecall = " + str(recall))
        print("\nF-score = " + str(fscore))

    elif num_folds > 1:
        X, Y, _ = data_io.load_tensor_data(data_id)
        print("Running a k-fold analysis with " + str(num_folds) + " fold(s).")
        kfold_results = fk.run_kfold(X, Y, n_splits=num_folds, epoch_cnt=epoch_num, batch_size=batch_size, model_out=model_out)

        with open(data_id + ".kfold_results.json", 'w') as file_out:
            file_out.write(json.dumps(kfold_results, default=default))

        fk.summarize_kfold(kfold_results, drop_minmax=True)
    else:
        raise ValueError("The number of folds is invalid. Must be > 0.")


# Use a trained ML model to evaluate beamforming results for a detection
@click.command('detect', short_help="Apply a model to detect signals")
@click.option("--model-id", help="Model name (required)", prompt="Relative/path/to/model")
@click.option("--fk-file", help="Name of fk beamforming results (required)", prompt="Relative/path/to/fk/results/file")
def detect(model_id: str, fk_file: str) -> None:
    '''
    \b
    riddl models fk detect
    ----------------------
    Args:
        model_id (str): Relative/path/to/model
        fk_file (str): Relative/path/to/fk/results/file
    \b
    Example Usage:
    (from the `examples' folder)
    \t riddl models fk detect --model-id ./models/use/soi2 --fk-file ./data/fk_array_data/IM.IS53_2022.03.01_08.00.00-08.20.24.fk_results.dat
    '''

    click.echo("")
    click.echo("#####################################")
    click.echo("##                                 ##")
    click.echo("##              riddl              ##")
    click.echo("##       AI/ML Model Methods       ##")
    click.echo("##                                 ##")
    click.echo("##    Apply an fk detection model  ##")
    click.echo("##    to infrasound array data.    ##")
    click.echo("##                                 ##")
    click.echo("#####################################")
    click.echo("")

    # Run analysis and summarize results
    det_times, predictions = fk.run(fk_file, model_id)
    for n in range(0, len(det_times)):
        print(det_times[n], '\t', predictions[n])

    fk.plot(fk_file, det_times, predictions)


# Use a trained ML model to evaluate beamforming results for a detection
@click.command('evaluate', short_help="Evaluate a model on a set of tensors")
@click.option("--model-id", help="Model name (required)", prompt="Relative/path/to/model")
@click.option("--data-id", help="Directory of atmospheric specifications (required)", prompt="prompt data_id")
@click.option("--batch_size", help="Directory of atmospheric specifications (required)", default=32)
def evaluate(model_id: str, data_id: str, batch_size: float) -> None:
    '''
    \b
    riddl models fk detect
    ----------------------
    Args:
        model_id (str): Relative/path/to/model
        fk_file (str): Relative/path/to/fk/results/file
    \b
    Example Usage:
    (from the `examples' folder)
    \t riddl models fk evaluate --model-id ./I57_transportability --data-id I57
    '''

    click.echo("")
    click.echo("#####################################")
    click.echo("##                                 ##")
    click.echo("##              riddl              ##")
    click.echo("##       AI/ML Model Methods       ##")
    click.echo("##                                 ##")
    click.echo("##    Evaluate a trained ML model  ##")
    click.echo("##      to infrasound data set.    ##")
    click.echo("##                                 ##")
    click.echo("#####################################")
    click.echo("")

    X, Y, _ = data_io.load_tensor_data(data_id)
    
    model = load_model(model_id)
    score, M = fk.evaluate_model(X, Y, model=model, batch_size=batch_size, verbosity=1)
    m, n = np.shape(M)
    width = 12
    print("The loss is " + str(score[0]) + ". The accuracy is " + str(score[1]) + ".")
    print("The confusion matrix is ")
    print((n*(width+1)+1)*"~")
    for jj in range(0, m):
        line_temp = "|"
        for ii in range(0, n):
            line_temp = line_temp + '{message:{fill}{align}{width}}'.format(message=str(M[jj, ii]), fill=' ', align='^', width=width,) + "|"
        print(line_temp) 
    print((n*(width+1)+1)*"~")
