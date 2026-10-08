# acc-bench user guide

This guide tells you how to use acc-bench. It is written in ASD-STE100 Simplified
Technical English. The test `tests/test_user_guide.py` examines the guide for the STE
rules that a script can examine. The same test runs all the example files in this guide.

## 1 About acc-bench

acc-bench measures how well different methods predict a label on your data. It then
tells you which of your claims about the methods are correct for your data.

acc-bench has seven units. Each unit does one task and writes one file. Each unit also
stops your work if it finds a problem. The table shows the units in the order that you
use them.

| Unit | What the unit does | The file that the unit uses or writes |
|---|---|---|
| 0 Construct | Records the decision that each number helps you make | `construct.yaml` (you write this file) |
| 1 Fixtures | Divides your data into training, validation and test rows, and locks the division | `fixtures/<hash>/` |
| 2 Channels | Records the path of each column into a model | `channel_manifest.json` |
| 3 Runner | Trains and scores each method in each cell, and records each run | `runs/`, `runs.coverage.json` and `runs.history.jsonl` |
| 4 Resolution | Calculates the smallest difference that each cell can show | `resolution.json` |
| 5 Rules | Gives one verdict for each question that you wrote before the runs | `verdicts.json` and `null_suite.json` |
| 6 Claims | Records each claim before the runs, and its result after the runs | `claims.jsonl` |

Each file records the state of the code and the data that made it. If the code or the
data changes, acc-bench refuses to use the old file. You must make the file again.

## 2 Words in this guide

| Word | What it means |
|---|---|
| Project folder | The folder that contains all the files for one benchmark. Run all commands in this folder. Put it outside the `acc-bench` folder. |
| Task | One label that you predict, for example readmission. A task has a name, for example `readmitted`. |
| Target | The definition of the label of a task. |
| Cell | One part of the data of a task, for example the patients of one hospital. |
| Cell name | `<task>/<value>`, for example `readmitted/south`. If a task has only one cell, the cell name is the task name. |
| Method | A model that acc-bench trains and scores, for example `logreg`. |
| Run | One method, trained and scored on one cell, with one split key and one fit seed. |
| Split key | A name that sets one division of the training and validation rows. The test rows are the same for all split keys. |
| Fit seed | A number that sets the random start of a model. |
| Noise floor | The smallest difference between two scores that a cell can show. A smaller difference can be noise. |
| Question | A question that one decision rule answers after the runs. |
| Verdict | The answer of a decision rule: `effect`, `no_effect` or `inconclusive`. |
| Claim | A statement that a question decides, for example "Method A is better in the south hospital". |

## 3 Before you start

Make sure that you have these items:

1. Python 3.10 or a later version.
2. Git.
3. Your data in one CSV file.

## 4 Install acc-bench

1. Open a terminal.
2. Copy the repository to your computer. Run this command. If the repository is private,
   you must have access to it.

   ```bash
   git clone https://github.com/Accendero/acc-bench.git
   ```

3. Go into the folder that the command made:

   ```bash
   cd acc-bench
   ```

4. Make a virtual environment for acc-bench. Run this command:

   ```bash
   python -m venv .venv
   ```

5. Start the virtual environment. Use the command for your terminal:

   | Terminal | Command |
   |---|---|
   | Windows Command Prompt | `.venv\Scripts\activate` |
   | Windows PowerShell | `.venv\Scripts\Activate.ps1` |
   | Git Bash on Windows | `source .venv/Scripts/activate` |
   | Linux or macOS | `source .venv/bin/activate` |

6. Install acc-bench. Run this command:

   ```bash
   python -m pip install .
   ```

7. Make sure that the installation is correct. Run this command:

   ```bash
   acc-bench --version
   ```

   The terminal shows `acc-bench` and a version number, for example `acc-bench 0.1.0` or
   `acc-bench 0.1.0.dev0`.

NOTE: Start the virtual environment again each time that you open a new terminal. If
PowerShell does not let the script start, refer to the Microsoft information about
execution policies.

## 5 Run the example

The example uses synthetic data. It does all the steps in section 6 for you. Use it to
make sure that acc-bench operates correctly on your computer.

1. Go to the `acc-bench` folder.
2. Run this command:

   ```bash
   python examples/synthetic/run_example.py
   ```

3. Wait approximately 30 seconds.
4. Make sure that the last lines of the output show the seven units, with a file for each
   unit.
5. Find the files in the folder `examples/synthetic/out/`.

NOTE: The example for real clinical-trial data is in `examples/trialbench/`. That
example downloads 29.4 MB of data. Refer to `examples/trialbench/README.md`.

## 6 Make your own benchmark

Do the steps in this section in the given sequence. Each step uses the files from the
steps before it.

This section uses one example. The data is about patients in two hospitals. The task is
to predict if a patient comes back to hospital. All the example files in this section
are complete. You can copy them and change them for your data.

Each command reads its files from the project folder. The table gives the file names
that each command reads. Use these names.

| Command | Reads |
|---|---|
| `acc-bench construct check` | `construct.yaml` |
| `acc-bench fixtures freeze` | `fixtures.yaml`, then the files that it names |
| `acc-bench channels check` | `channels.yaml` and `fixtures.yaml` |
| `acc-bench rules test questions.yaml` | `questions.yaml` and `grid.yaml` |
| `acc-bench claims register` | `questions.yaml` and `construct.yaml` |
| `acc-bench run` | `grid.yaml`, then the files that it names |
| `acc-bench resolve` | `grid.yaml` |
| `acc-bench verdict` | `questions.yaml` |
| `acc-bench claims resolve` | `verdicts.json` and `claims.jsonl` |

### 6.1 Prepare your data

1. Make a project folder. Put it outside the `acc-bench` folder.
2. Put your data in one CSV file in the project folder, for example
   `data/patients.csv`.
3. Make sure that each row is one item, for example one patient.
4. Make sure that one column has a different identifier for each row.
5. Make sure that one column gives the label as `0` or `1`. For more than two classes,
   use `0`, `1`, `2` and the subsequent numbers.
6. If your data has more than one cell, make sure that one column gives the cell of each
   row.

The example data has these columns:

| Column | Contents |
|---|---|
| `patient_id` | The identifier, for example `N00001` |
| `hospital` | The cell: `north` or `south` |
| `age` | A number |
| `length_of_stay` | A number of days |
| `ward` | A category: `cardiology`, `surgery` or `general` |
| `discharge_note` | Free text |
| `readmitted` | The label: `1` if the patient came back, `0` if not |

### 6.2 Write the construct statement

The construct statement records the decision that your benchmark helps you make. Write
it before you do the other steps.

1. Make a file with the name `construct.yaml` in the project folder.
2. Write the fields in the example below. Each field is necessary.
3. Under `tasks`, write a list. Give each task a `name` and a `reason`.
4. Give the target the same name as the task.
5. For `metric`, use one of these names: `pr_auc`, `roc_auc`, `accuracy`,
   `balanced_accuracy`, `f1`, `macro_f1`, `brier` or `log_loss`.
6. Examine the file. Run this command:

   ```bash
   acc-bench construct check
   ```

7. If the command shows problems, correct each problem. Then do step 6 again.

Example `construct.yaml`:

```yaml
version: 1
stated_on: 2026-10-07
owner: A. Person
question: Which method should rank patients for a follow-up call after discharge?
decision: Which patients a nurse calls first.
target:
  name: readmitted
  definition: The patient came back to hospital within 30 days.
metric:
  name: pr_auc
  reason: The nurse calls only the patients at the top of the list.
tasks:
  - name: readmitted
    reason: The call list is for readmission only.
reported_numbers:
  - name: pr_auc_by_hospital
    decision: Which method to use in each hospital.
```

The fields are:

| Field | What to write |
|---|---|
| `version` | `1`. Increase it by 1 for each change to the statement. Refer to section 7.3. |
| `stated_on` | The date on which you write the statement |
| `owner` | The name of the person who is responsible for the benchmark. Free text. |
| `question` | The question that the benchmark answers. Free text. |
| `decision` | The decision that the answer helps you make. Free text. |
| `target` | The `name` of the label, which is the task name, and its `definition` |
| `metric` | The `name` of the metric, and the `reason` for it |
| `tasks` | For each task, the `name` and the `reason` |
| `reported_numbers` | For each number in your report, a `name` and the `decision` that it helps you make. The name is a free label. |

You can also write two optional fields. Under `metric`, `alternatives` gives the other
metrics that you did not use, and why. Under `measures`, you can record a measure that
you selected in place of a different measure. Refer to `examples/synthetic/construct.yaml`.

### 6.3 Freeze the fixtures

The fixture divides each cell into training, validation and test rows. After you freeze
the fixture, the division cannot change.

1. Make a file with the name `fixtures.yaml` in the project folder.
2. Write the fields in the example below.
3. Under `tasks`, write each task name, with its fields under it. Each task name must be
   a task name in `construct.yaml`.
4. For `label_rule`, write `{column: <name>}` if one column gives the label.
5. For `cell_by`, write the name of the column that gives the cell. If the task has only
   one cell, do not write `cell_by`.
6. Freeze the fixture. Run this command:

   ```bash
   acc-bench fixtures freeze
   ```

7. Make sure that the output shows each cell, with the number of training, validation
   and test rows. The numbers are for the first split key.

Example `fixtures.yaml`:

```yaml
construct: construct.yaml
id_column: patient_id
tasks:
  readmitted:
    source: data/patients.csv
    type: binary
    label_rule: {column: readmitted}
    cell_by: hospital
split:
  test: {fraction: 0.25, key: test-v1}
  validation: {fraction: 0.2}
  keys: [split-0, split-1]
```

The example puts 25 percent of the rows in the test set. Then it puts 20 percent of the
other rows in the validation set. The `key` of the test set is a name that sets which
rows go to the test set. A different key gives different test rows.

If your data already gives the test rows, write `test: {column: <name>, value: <value>}`
in place of the fraction and the key. For example, `test: {column: split, value: test}` puts each row
with `test` in the column `split` in the test set.

If a Python function calculates the label, do these steps:

1. Write the function in a Python file in the project folder, for example `labels.py`.
2. Make the function give one label, `0` or `1`, for each row of the table, in the same
   sequence. The input of the function is the table, as a pandas DataFrame.
3. In `fixtures.yaml`, write `label_rule: labels:failed`. Use the file name without
   `.py`, then the name of the function.
4. Under `label_columns`, write each column that the function reads.

acc-bench calculates the labels again from the columns in `label_columns` only. If the
labels are different, the freeze stops. A column in `label_columns` must not go to a
model, because it gives the label.

Example of a complete `fixtures.yaml` with a label function. The data has the columns
`serial`, `site` and `failure_score`. The label is `1` if `failure_score` is more than
1.0:

```yaml
construct: construct.yaml
id_column: serial
tasks:
  failed:
    source: data/devices.csv
    type: binary
    label_rule: labels:failed
    label_columns: [failure_score]
    cell_by: site
split:
  test: {fraction: 0.25, key: test-v1}
  validation: {fraction: 0.2}
  keys: [split-0, split-1]
```

```python
def failed(table):
    return (table["failure_score"] > 1.0).astype(int)
```

Each name in `split.keys` gives a different division of the training and validation
rows. To measure how much the division changes the scores, use two or more names.

### 6.4 Register the columns

Each column in your data must go to one channel, or you must ignore it. A channel is a
group of columns that a model reads in the same way.

1. Make a file with the name `channels.yaml` in the project folder.
2. Under `channels`, write each channel. Give each channel a `kind` and its `columns`.
3. Under `ignore`, write each column that no model must read. Give the reason for each
   column.
4. Put the identifier column under `ignore`. If one column gives the label, put it under
   `ignore`. If a function gives the label, put each column in `label_columns` under
   `ignore`.
5. Put the cell column under `ignore`. In one cell, all the rows have the same value in
   this column. Thus the column gives no information to a model.
6. Put each column that `fixtures.yaml` uses for the test rows under `ignore`.
7. Examine the registry. Run this command:

   ```bash
   acc-bench channels check
   ```

8. If the command shows a column that is not registered, add the column to a channel or
   to `ignore`. Then do step 7 again.

Use one of these kinds:

| Kind | For columns that contain | Settings, with the default value |
|---|---|---|
| `numeric` | Numbers | Optional: `impute` (`median`, or `mean` or `zero`), `scale` (`true`), `missing_indicator` (`false`) |
| `categorical` | Categories | Necessary: `encoding` (`one_hot` or `target`). Optional: `min_frequency` (`1`) |
| `text` | Free text | Optional: `max_features` (`20000`), `ngram_max` (`1`), `min_df` (`1`) |
| `multihot` | Lists of codes, for example `I10;E11`. An empty value is permitted. | Optional: `separator` (`;`), `min_df` (`1`) |

A column of grades, for example `A` to `E`, is categories. Use `categorical` with
`one_hot`. If you change the grades to numbers before step 1, use `numeric`.

Example `channels.yaml`:

```yaml
channels:
  measurements:
    kind: numeric
    columns: [age, length_of_stay]
  ward:
    kind: categorical
    encoding: one_hot
    columns: [ward]
  notes:
    kind: text
    columns: [discharge_note]
ignore:
  patient_id: the identifier
  hospital: the cell column
  readmitted: the label
```

The command finds the data through `fixtures.yaml`. It writes `channel_manifest.json`.

### 6.5 Write the grid

The grid gives the methods and the runs.

1. Make a file with the name `grid.yaml` in the project folder.
2. Write the fields in the example below.
3. Under `methods`, write one or more of these methods: `majority`, `logreg`, `hist_gbm`
   and `tfidf_logreg`. `tfidf_logreg` reads only `text` channels. `logreg` and `hist_gbm`
   read the `numeric`, `categorical` and `multihot` channels. `majority` reads no channel.
4. Under `split_keys`, write names from `split.keys` in `fixtures.yaml`.
5. Under `threads`, write the number of processor threads that each run can use. Use
   `1` if you are not sure. Some methods give different results with a different number
   of threads.

Example `grid.yaml`:

```yaml
construct: construct.yaml
fixture: fixtures.yaml
registry: channels.yaml
channel_manifest: channel_manifest.json
methods: [majority, logreg, hist_gbm, tfidf_logreg]
split_keys: [split-0, split-1]
fit_seeds: [0]
threads: 1
out: runs
```

Do not start the runs yet. Write the questions and the claims first.

### 6.6 Write the questions

Each question has one decision rule. The rule gives the verdict after the runs. Write
the questions before the runs.

1. Make a file with the name `questions.yaml` in the project folder.
2. Write the fields in the example below. The file `resolution.json` does not exist yet.
   acc-bench writes it in section 6.9.
3. Give each question an `id`, the `question` text, a `rule` and its `params`.
4. Use the cell names from the output of section 6.3, for example `readmitted/south`.
5. Examine the questions. Run this command:

   ```bash
   acc-bench rules test questions.yaml
   ```

6. Make sure that the output shows `match the grid` and `pass` for each rule. A rule
   passes if it finds an effect in 3 or fewer of 20 sets of data with no effect.

Use one of these rules:

| Rule | The question that it answers | Parameters |
|---|---|---|
| `difference_clears_floor` | Is there a difference between method `a` and method `b` in one cell? The verdict names the better method. | `cell`, `a`, `b` |
| `k_of_n_cells` | Is method `a` better than method `b` in `k` or more of the cells? Only a better `a` counts. | `cells` (a list), `a`, `b`, `k` |
| `difference_of_drops` | Is the difference between two methods larger in one arm than in a second arm? | `cell`, `model`, `reference`, `seen_arm`, `unseen_arm` |

For `a`, `b`, `model` or `reference`, write a method name. acc-bench can also select the
best method from a list. To do this, write `{select: [<method>, <method>]}`.

acc-bench selects the method with the best mean validation score. It uses the metric in
`construct.yaml` and all the split keys and fit seeds. It does not use the test rows.
The output of section 6.10 shows the selected method.

Example `questions.yaml`:

```yaml
grid: grid.yaml
resolution: resolution.json
questions:
  - id: Q1
    question: Is the text model better than the best tabular model in the south hospital?
    rule: difference_clears_floor
    params:
      cell: readmitted/south
      a: tfidf_logreg
      b: {select: [logreg, hist_gbm]}
  - id: Q2
    question: Is the text model better than logreg in both hospitals?
    rule: k_of_n_cells
    params:
      cells: [readmitted/north, readmitted/south]
      a: tfidf_logreg
      b: logreg
      k: 2
```

### 6.7 Register the claims

A claim is a statement that one question decides.

WARNING: Register each claim before you start the runs in section 6.8. If you register a
new claim after the first run, acc-bench cannot resolve the claim.

1. For each claim, run the command `acc-bench claims register`. The example registers
   two claims:

   ```bash
   acc-bench claims register C1 --statement "The text model is better in the south hospital." --question Q1 --prediction effect:tfidf_logreg
   acc-bench claims register C2 --statement "The text model is not better in both hospitals." --question Q2 --prediction no_effect
   ```

2. For `--prediction`, use one of these values:

   | Value | What it means | Use it with |
   |---|---|---|
   | `effect` | The rule will find an effect | All rules |
   | `effect:a` or `effect:b` | The rule will find that the method in `a`, or in `b`, is better | `difference_clears_floor` |
   | `effect:<method>` | The rule will find that this method is better | `difference_clears_floor` |
   | `no_effect` | The rule will find no effect | `difference_clears_floor` and `k_of_n_cells` |

   For `difference_clears_floor`, give the direction. With `effect` only, a result in
   either direction confirms the claim. Write `effect:a` if you predict that the method in
   `a` is better. Write `effect:b` for the method in `b`.

   Use `effect:a` or `effect:b` if a side has a `select` list. You cannot know the selected
   method before the runs. The verdict names the better method. acc-bench compares that
   method with the method that it used for `a` or `b`.

   acc-bench refuses a direction for a rule that does not report one.

3. If you will not test a claim, register it with the question that is nearest to it.
   acc-bench records the question and the prediction, but does not use them. Then record
   the reason. Run these commands:

   ```bash
   acc-bench claims register C3 --statement "Older patients come back more often." --question Q1 --prediction effect
   acc-bench claims untested C3 --reason "The data has no age groups for this test."
   ```

4. Use the option `--exploratory` for a claim that you did not plan before the study. Only
   an exploratory claim can use a verdict from `oracle_select`.

### 6.8 Run the methods

1. Start the runs. Run this command:

   ```bash
   acc-bench run
   ```

2. Make sure that the last line shows `error 0`, `missing 0` and `stale 0`.
3. If a run stops before the end, run the same command again. acc-bench continues from
   the last complete run.

### 6.9 Calculate the noise floors

1. Run this command:

   ```bash
   acc-bench resolve
   ```

2. Read the table that the command shows. Each row is one cell.

The columns of the table are:

| Column | What it means |
|---|---|
| `floor` | The noise floor of the cell. It is the largest of the three terms. |
| `source` | The term that sets the floor: `split`, `fit` or `test` |
| `split` | The noise from a change of split key |
| `fit` | The noise from a change of fit seed. `--` shows that the grid has only one fit seed. `0.0000` shows that the methods give the same result for each seed. |
| `test` | The noise from the sample of test rows |

The text `cannot resolve 0.05` shows that the floor of the cell is more than 0.05. In
that cell, do not report a difference smaller than the floor. To change the value 0.05,
use the option `--threshold`.

If the `test` term sets the floor, more test rows make the floor smaller. More split
keys or fit seeds do not make it smaller.

### 6.10 Find the verdicts

1. Run this command:

   ```bash
   acc-bench verdict
   ```

2. Read the verdict for each question.

Before acc-bench writes a verdict, it tests each rule on data that has no effect. The
first line of the output shows the result of this test. If a rule shows an effect on
that data, acc-bench does not write a verdict.

| Verdict | What it means |
|---|---|
| `effect` | The data shows the effect. The difference is larger than the noise floor. |
| `no_effect` | The data shows that the effect is not there, or that it cannot reach the rule. |
| `inconclusive` | The data cannot decide. Usually the noise floor is too large. |

### 6.11 Resolve the claims

1. Run this command:

   ```bash
   acc-bench claims resolve
   ```

2. Show the register. Run this command:

   ```bash
   acc-bench claims show
   ```

3. Report all claims, not only the claims that are confirmed.

Each claim has one of these results:

| Result | What it means |
|---|---|
| `confirmed` | The verdict agrees with the prediction |
| `refuted` | The verdict does not agree with the prediction |
| `inverted` | The verdict found an effect, but the other method is better |
| `inconclusive` | The verdict is `inconclusive` |
| `untested` | You recorded that you will not test the claim |

### 6.12 Examine your report

1. In your report, write `[claim:<id>]` after each statement that comes from a claim.
   For example: "The text model is better in the south hospital [claim:C1]."
2. Write what the data shows, not what the claim predicted. For an `inverted` claim, the
   data shows the opposite of the prediction. Write that result, and cite the claim.
   For an `untested` claim, write that you did not test it, and why.
3. Examine the report. Run this command:

   ```bash
   acc-bench claims check-doc report.md
   ```

4. The command can show a claim that is not in the register, or a claim with no current
   result. If it does, correct the report, or do the step that the message gives.
5. To make sure that the report cites all the claims, add the option `--require-all`:

   ```bash
   acc-bench claims check-doc report.md --require-all
   ```

6. Read each statement again. Make sure that it agrees with the result that the command
   shows for its claim.

## 7 Make changes after the runs

### 7.1 Change the code

When you change the code, acc-bench does not use the old runs. This prevents results
from old code and new code in one table. After a change to the code, the command
`acc-bench run` stops with a message that contains `are stale`.

The code is all the Python files in the project folder and its subfolders, the files
`pyproject.toml` and `requirements*.txt`, and the code of acc-bench. A new Python file, or a
change to a comment, is also a change to the code.

1. Change the code.
2. Do the runs again. Run this command:

   ```bash
   acc-bench run --rerun-stale
   ```

3. Do sections 6.9 to 6.12 again.

### 7.2 Change the data or a configuration file

1. Change the data or the file.
2. Freeze the fixture again. Run this command:

   ```bash
   acc-bench fixtures freeze
   ```

3. Examine the registry again. Run this command:

   ```bash
   acc-bench channels check
   ```

4. Do the runs again. Run this command:

   ```bash
   acc-bench run --rerun-stale
   ```

5. Do sections 6.9 to 6.12 again. Do not register the claims again.

NOTE: acc-bench does not delete the old fixture folders in `fixtures/`. You can delete a
folder if no file in the project folder refers to it.

### 7.3 Change a claim

CAUTION: Do not change a claim without a record. acc-bench keeps each change in the
register, with its date and its reason.

A change to a claim is also a change to the construct statement. Thus the fixture and
the runs change too. Section 6.6 step 5 finds errors in the questions before you
register the claims. This prevents a change to a claim because of an error in a
question.

You can change the question, the prediction and the statement of a claim. Do the steps
in this sequence.

1. In `construct.yaml`, increase `version` by 1.
2. Add the key `amendments` at the top level of the file, if it is not there. Under it,
   add the new `version`, the `date` of the change and the `reason`. For example:

   ```yaml
   version: 2
   amendments:
     - version: 2
       date: 2026-10-07
       reason: Q2 is the correct question for claim C1.
   ```

3. Change the claim. Give each item that changes: `--question`, `--prediction` or
   `--statement`. If the new question uses a different rule, also give a new prediction.
   If the statement does not agree with the new question, give a new statement. Run this
   command:

   ```bash
   acc-bench claims amend C1 --reason "Q2 is the correct question." --question Q2 --prediction no_effect --statement "The text model is not better in both hospitals."
   ```

4. If acc-bench refuses the prediction, give a prediction that the new question can
   decide. Refer to the table in section 6.7.
5. Change the report so that it agrees with the new claim.
6. Do steps 2 to 5 of section 7.2. These steps include section 6.12.

After a change, `acc-bench claims show` writes `(amended)` after the claim. If you
changed the claim after the first run, it writes `(amended after results)` after
`acc-bench claims resolve`. Tell the readers of your report about this change.

## 8 Add your own method

1. Make a Python file in the project folder, for example `methods.py`.
2. In the file, write a subclass of `accbench.methods.SklearnMethod`.
3. Give the subclass a `name` and a `make` function. The `make` function gives a
   scikit-learn model.
4. At the end of the file, call `register` with the subclass.
5. In `grid.yaml`, write `methods_module: methods.py`.
6. Add the method name to `methods` in `grid.yaml`.

Refer to `examples/synthetic/methods.py` for a complete file.

NOTE: If your method uses a model with a price for each call, write the price in a
price table. Refer to `examples/synthetic/prices.yaml`. Without a price, acc-bench stops
the run and records an error.

## 9 Add your own decision rule

1. Make a Python file in the project folder, for example `my_rules.py`.
2. Write a function that gives a `Verdict`. Put the decorator `@rule` on it.
3. Give the decorator a `null_input` function. This function makes data that has no
   effect. Refer to the example below.
4. In `questions.yaml`, write `rules_module: my_rules.py`.
5. Examine the rule. Run this command:

   ```bash
   acc-bench rules test questions.yaml
   ```

6. Make sure that the rule shows `pass`. If it shows `FAIL`, the rule finds effects that
   are not there. Correct the rule.

This example rule compares the scores of two methods in one cell. The null input gives
two methods with the same skill.

```python
import numpy as np

from accbench.rules import SyntheticContext, Verdict, rule


def same_skill(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 300)
    ids = np.array([f"t{i}" for i in range(300)])
    rows = {
        ("c", "a"): (y, y + rng.normal(0, 1, 300), ids, None),
        ("c", "b"): (y, y + rng.normal(0, 1, 300), ids, None),
    }
    return SyntheticContext(rows, {"c": 0.05}), {"cell": "c", "a": "a", "b": "b"}


@rule("a_beats_b_by_the_floor", null_input=same_skill)
def a_beats_b_by_the_floor(ctx, *, cell, a, b):
    y, score_a, _ = ctx.test_rows(cell, a)
    _, score_b, _ = ctx.test_rows(cell, b)
    difference = ctx.goodness(y, score_a) - ctx.goodness(y, score_b)
    if difference > 2 * ctx.floor(cell):
        return Verdict("effect", f"{a} is better by {difference:.3f}")
    return Verdict("inconclusive", f"difference {difference:+.3f}")
```

## 10 Problems and their solutions

| The message contains | Cause | Action |
|---|---|---|
| `state the construct before freezing` | `construct.yaml` is not in the project folder | Do section 6.2 |
| `must map at least one task name` | In `fixtures.yaml`, `tasks` is a list | Write each task name with its fields under it. Refer to section 6.3. |
| `not in the construct statement` | A task in `fixtures.yaml` is not in `construct.yaml` | Use the same task names in the two files |
| `appears more than once` | Two rows have the same identifier | Make each identifier different |
| `must name the columns it reads` | A label function has no `label_columns` | Add `label_columns`. Refer to section 6.3. |
| `label_columns does not name` | The label function reads a column that is not in `label_columns` | Add the column to `label_columns` |
| `has one class only` | All the rows of a split in a cell have the same label | Use a larger test fraction, or put more rows in the cell |
| `has not been frozen` | The data, the construct or `fixtures.yaml` changed after the freeze | Run `acc-bench fixtures freeze` |
| `modified after it was frozen` | A file in `fixtures/` changed | Run `acc-bench fixtures freeze` |
| `is not registered to any channel` | A column is not in `channels.yaml` | Add the column to a channel or to `ignore` |
| `which the table lacks` | `channels.yaml` has a column name that is not in the data | Correct the column name |
| `would leak` | The label column is in a channel | Move the label column to `ignore` |
| `freeze it again` | `construct.yaml` changed after the freeze | Do steps 2 to 5 of section 7.2 |
| `does not exist yet` | `questions.yaml` names a grid that is not there | Do section 6.5 |
| `is not a cell` | A question has a cell name that is not correct | Use a cell name from the message |
| `needs parameter` or `has no parameter` | A question has incorrect parameters | Use the parameters in the table in section 6.6 |
| `is not in the grid's methods` | A question names a method that is not in `grid.yaml` | Add the method to `grid.yaml`, or correct the name |
| `are stale` | The code changed after the runs | Do section 7.1 |
| `is stale, recompute it` | A file that this file uses changed | Do the step that writes the file again |
| `the code changed` | The code changed after acc-bench wrote the file | Do section 7.1 |
| `are not complete` and `stale` more than 0 | The code changed after the runs | Do section 7.1 |
| `are not complete` and `stale 0` | One or more runs failed, or have no record | Run `acc-bench coverage`, correct the problem, then run `acc-bench run` |
| `its verdicts are out of date` | The code changed after the verdicts | Do section 7.1 |
| `the verdicts changed after it was resolved` | You made the verdicts again, but did not resolve the claims | Run `acc-bench claims resolve` |
| `fail the null-input test` | A decision rule finds effects that are not there | Correct the rule. Refer to section 9. |
| `after the first run` | The claim was registered after the runs | Register a new claim before new runs. Refer to section 6.7. |
| `claim ... is not registered` | The claim is not in the register | Register the claim first. Refer to section 6.7. |
| `does not report which method is better` | The prediction gives a direction, but the rule cannot decide a direction | Use `effect` or `no_effect` |
| `which is not a or b` | The prediction names a method that is not in the question | Use `effect:a`, `effect:b`, or a method of the question |
| `does not cite these registered claims` | With `--require-all`, the report does not cite each claim | Cite each claim in the report |
| `has changed since the claim was registered` | The question changed after you registered the claim | Do section 7.3 |
| `oracle selection` | The question selects a method with the test rows | Use `select` in place of `oracle_select` |
| `construct statement amended first` | The claim changed, but `construct.yaml` did not | Do section 7.3, steps 1 and 2 |
| `not in the price table` | A method used a model without a price | Add the price. Refer to section 8. |

If the terminal shows a Python `Traceback` in place of a message, there is an error in
acc-bench or in your own code. Report the full output to the people who maintain
acc-bench.

## 11 Commands

| Command | What it does |
|---|---|
| `acc-bench construct check` | Examines `construct.yaml` |
| `acc-bench fixtures freeze` | Freezes the fixture that `fixtures.yaml` describes |
| `acc-bench fixtures show <folder>` | Shows what a fixture contains and when it was frozen |
| `acc-bench channels check` | Examines the data against `channels.yaml` and writes `channel_manifest.json` |
| `acc-bench run` | Does each run that has no record |
| `acc-bench coverage` | Counts the runs in each status: `ok`, `skipped`, `error`, `missing` and `stale` |
| `acc-bench resolve` | Calculates the noise floor of each cell |
| `acc-bench rules test` | Examines the questions, and tests each decision rule on data that has no effect |
| `acc-bench verdict` | Gives the verdict for each question |
| `acc-bench claims register` | Registers a claim |
| `acc-bench claims untested` | Records that a claim will not be tested, and why |
| `acc-bench claims amend` | Changes a claim, with a date and a reason |
| `acc-bench claims resolve` | Gives each claim its result |
| `acc-bench claims show` | Shows the register |
| `acc-bench claims check-doc <file>` | Examines the claims that a report cites, and shows the claims that it does not cite |

To see all options of a command, add `--help`, for example `acc-bench run --help`.
