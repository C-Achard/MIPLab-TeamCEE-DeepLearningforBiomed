"""Functions for training and evaluating models."""
from sklearn.model_selection import train_test_split


def balanced_data_shuffle(dataset_dataframe, test_size=0.2):
    """Shuffle and stratify the data by task, so that each task is represented equally in the train and test sets.

    Also ensures no subject is present only in the test set.
    If this is the case, half of the tasks of this subject are moved to the train set.
    """
    train_subjects, test_subjects = train_test_split(
        dataset_dataframe,
        test_size=test_size,
        stratify=dataset_dataframe["task"],
    )
    # find if subjects are present only in the test set
    test_only_subjects = test_subjects[
        ~test_subjects["subject"].isin(train_subjects["subject"])
    ]
    if len(test_only_subjects) > 0:
        print(
            f"Found {len(test_only_subjects['subject'].unique())} subjects present only in the test set"
        )
        # if there are subjects present only in the test set, move half of their tasks to the train set
        for subject in test_only_subjects["subject"].unique():
            subject_tasks = test_subjects[
                test_subjects["subject"] == subject
            ].sample(frac=0.5)
            train_subjects = train_subjects.append(subject_tasks)
            test_subjects = test_subjects.drop(subject_tasks.index)
            print(
                f"Moved {len(subject_tasks)} tasks from subject {subject} to the train set"
            )
